package com.haven.chat

/** One search result, small enough to show to Gemma. */
data class SearchHit(
    val id: Long,
    val date: String,
    val question: String,
    val answer: String,
    val tags: String,
    /** Reranker score: higher = more relevant. */
    val score: Float
)

/** Something went wrong in the search pipeline. The message says what, in plain words. */
class SearchException(message: String) : Exception(message)

/**
 * The two search models the pipeline needs. (An interface, so the pipeline can also be
 * tested without a phone.) Only one model is loaded at a time.
 */
interface SearchModels {
    /** Loads a search model. Returns 0 = ok, otherwise an error code (see LlamaBridge.loadAux). */
    fun load(path: String, kind: Int, threads: Int): Int
    fun embed(text: String): FloatArray?
    fun rerank(query: String, doc: String): Float
    fun unload()
    fun lastLog(): String
}

/** SearchModels on top of the native engine (LlamaBridge). */
class BridgeSearchModels(private val b: LlamaBridge) : SearchModels {
    override fun load(path: String, kind: Int, threads: Int): Int = b.loadAux(path, kind, threads)
    override fun embed(text: String): FloatArray? = b.embed(text.toByteArray(Charsets.UTF_8))
    override fun rerank(query: String, doc: String): Float =
        b.rerankScore(query.toByteArray(Charsets.UTF_8), doc.toByteArray(Charsets.UTF_8))
    override fun unload() = b.unloadAux()
    override fun lastLog(): String = b.lastLog()
}

/**
 * The heavy search, like retrieval.py:
 *   1. load the embedder, give vectors to new memories, embed the question, unload it
 *   2. candidates = nearest vectors + a few keyword hits
 *   3. load the reranker, score every candidate against the question, unload it
 *   4. keep the best few
 *
 * IMPORTANT: the main model (Gemma) must be unloaded before search() or embedPending()
 * is called (RAM). Call from a background thread.
 */
class Retrieval(
    private val models: SearchModels,
    private val store: MemoryStore,
    private val embedderPath: String,
    private val rerankerPath: String,
    private val threads: Int,
    /** Progress messages, for example for the screen. */
    private val log: (String) -> Unit = {}
) {

    companion object {
        // The Python version used 50 + 10. Every candidate costs one reranker run on the phone,
        // so the default is smaller. Raise these when the speed is good enough.
        const val DEFAULT_VECTOR_CANDIDATES = 15
        const val DEFAULT_KEYWORD_CANDIDATES = 5
    }

    // ---------------------------------------------------------------- embedding new memories

    /** Gives a vector to every memory that has none. Returns how many were embedded. */
    fun embedPending(): Int {
        val rows = store.pending()
        if (rows.isEmpty()) return 0
        val t0 = System.nanoTime()
        withModel(embedderPath, LlamaBridge.KIND_EMBEDDER, "Embedder") {
            embedRows(rows)
        }
        log("embedded " + rows.size + " memories in " + secs(t0) + " s")
        return rows.size
    }

    private fun embedRows(rows: List<PendingRow>) {
        var done = 0
        for (chunk in rows.chunked(16)) {
            val pairs = ArrayList<Pair<Long, FloatArray>>()
            for (r in chunk) {
                val v = models.embed(embedText(r.tags, r.question, r.answer))
                    ?: throw SearchException(
                        "Embedding failed for memory #" + r.id + ". " + models.lastLog().takeLast(300)
                    )
                pairs.add(Pair(r.id, v))
            }
            store.setVecs(pairs)     // saved chunk by chunk, so work is not lost if something stops
            done += chunk.size
        }
    }

    // ---------------------------------------------------------------- search

    /**
     * Returns the best memories for a question (best first, at most topN).
     * keywords: extra words for the keyword search (synonyms from Gemma).
     * timeFilter: "last_week", "last_month", "last_year" or null.
     */
    fun search(
        query: String,
        keywords: List<String> = emptyList(),
        timeFilter: String? = null,
        topN: Int = MemoryConfig.TOP_N,
        vectorCandidates: Int = DEFAULT_VECTOR_CANDIDATES,
        keywordCandidates: Int = DEFAULT_KEYWORD_CANDIDATES
    ): List<SearchHit> {
        val q = query.trim()
        if (q.isEmpty() || store.count() == 0) return emptyList()

        // 1) Embedder: vectors for new memories, and the vector of the question.
        var t0 = System.nanoTime()
        var queryVec: FloatArray? = null
        withModel(embedderPath, LlamaBridge.KIND_EMBEDDER, "Embedder") {
            val pending = store.pending()
            if (pending.isNotEmpty()) embedRows(pending)
            queryVec = models.embed(q)
        }
        val qv = queryVec ?: throw SearchException("Embedding the question failed. " + models.lastLog().takeLast(300))
        log("embedder done in " + secs(t0) + " s")

        // 2) Candidates: nearest vectors first, then keyword hits that are not in the list yet.
        val ids = ArrayList<Long>(store.nearest(qv, timeFilter, vectorCandidates))
        for (id in store.hitIds(listOf(q) + keywords, timeFilter, keywordCandidates)) {
            if (id !in ids) ids.add(id)
        }
        val rows = store.getRows(ids)
        log(rows.size.toString() + " candidates")
        if (rows.isEmpty()) return emptyList()

        // 3) Reranker (not needed when there is only one candidate).
        val scored: List<Pair<MemoryRow, Float>>
        if (rows.size == 1) {
            scored = listOf(Pair(rows[0], 0f))
        } else {
            t0 = System.nanoTime()
            scored = withModel(rerankerPath, LlamaBridge.KIND_RERANKER, "Reranker") {
                rows.map { row ->
                    val score = models.rerank(q, embedText(row.tags, row.question, row.answer))
                    if (score <= -1e5f) {
                        throw SearchException("Reranking failed. " + models.lastLog().takeLast(300))
                    }
                    Pair(row, score)
                }
            }
            log("reranked " + rows.size + " candidates in " + secs(t0) + " s")
        }

        // 4) The best few, with a shortened answer.
        return scored
            .sortedByDescending { it.second }
            .take(topN)
            .map { (row, score) ->
                SearchHit(
                    row.id, row.eventDate, row.question,
                    row.answer.take(MemoryConfig.RESULT_ANSWER_CHARS), row.tags, score
                )
            }
    }

    // ---------------------------------------------------------------- helpers

    /** Loads one search model, runs block, and always unloads it again. */
    private fun <T> withModel(path: String, kind: Int, what: String, block: () -> T): T {
        val rc = models.load(path, kind, threads)
        if (rc != 0) throw SearchException(loadError(what, rc))
        try {
            return block()
        } finally {
            models.unload()
        }
    }

    private fun loadError(what: String, rc: Int): String {
        val reason = when (rc) {
            1 -> "the file could not be loaded"
            2 -> "not enough memory to start it"
            4 -> "the main model (Gemma) is still loaded; unload it first"
            else -> "error code $rc"
        }
        return "$what: $reason. " + models.lastLog().takeLast(300)
    }

    private fun secs(t0: Long): String = "%.1f".format((System.nanoTime() - t0) / 1e9)
}
