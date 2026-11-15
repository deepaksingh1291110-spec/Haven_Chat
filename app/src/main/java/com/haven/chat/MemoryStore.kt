package com.haven.chat

import android.database.sqlite.SQLiteException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.time.LocalDate
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter
import java.util.PriorityQueue

private val TIME_FORMAT = DateTimeFormatter.ofPattern("yyyy-MM-dd'T'HH:mm:ss")

/** Current time as text, for example 2026-11-14T10:22:33 (same as the Python memory.db). */
private fun nowIso(): String = LocalDateTime.now().format(TIME_FORMAT)

/** Today as text, for example 2026-11-14. */
private fun todayIso(): String = LocalDate.now().toString()

/** One saved memory, as read from the database. */
data class MemoryRow(
    val id: Long,
    val eventDate: String,
    val question: String,
    val answer: String,
    val tags: String
)

/** A memory that has no vector yet. */
data class PendingRow(
    val id: Long,
    val tags: String,
    val question: String,
    val answer: String
)

/**
 * Long-term memory: the raw question + raw answer + tags, as in store.py.
 * No summaries. Keyword search works the moment a row is saved; vectors are added later,
 * whenever the embedder is loaded anyway.
 * Call from a background thread.
 */
class MemoryStore(private val h: HavenDb) {

    private val db get() = h.db

    // ---------------------------------------------------------------- saving

    /**
     * Saves one memory and returns its id.
     * The same question + answer is never saved twice (the old id is returned).
     */
    fun add(question: String, answer: String, tags: List<String>): Long {
        val q = question.trim()
        val a = answer.trim().take(MemoryConfig.MAX_ANSWER_CHARS)
        val tagStr = tagText(tags)

        db.rawQuery("SELECT id FROM memories WHERE question = ? AND answer = ?", arrayOf(q, a)).use { c ->
            if (c.moveToFirst()) return c.getLong(0)
        }

        var newId = -1L
        db.beginTransaction()
        try {
            db.execSQL(
                "INSERT INTO memories(created_at, event_date, question, answer, tags) VALUES (?,?,?,?,?)",
                arrayOf<Any?>(nowIso(), todayIso(), q, a, tagStr)
            )
            db.rawQuery("SELECT last_insert_rowid()", null).use { c ->
                if (c.moveToFirst()) newId = c.getLong(0)
            }
            if (h.hasFts) {
                db.execSQL(
                    "INSERT INTO mem_fts(rowid, question, answer, tags) VALUES (?,?,?,?)",
                    arrayOf<Any?>(newId, q, a, tagStr)
                )
            }
            db.setTransactionSuccessful()
        } finally {
            db.endTransaction()
        }
        return newId
    }

    /** Deletes one memory (and its keyword index row). */
    fun delete(id: Long) {
        db.beginTransaction()
        try {
            db.execSQL("DELETE FROM memories WHERE id = ?", arrayOf<Any?>(id))
            if (h.hasFts) db.execSQL("DELETE FROM mem_fts WHERE rowid = ?", arrayOf<Any?>(id))
            db.setTransactionSuccessful()
        } finally {
            db.endTransaction()
        }
    }

    fun count(): Int = h.count("memories")

    // ---------------------------------------------------------------- keyword search (no model needed)

    /**
     * Ids of memories that contain any of the words (best match first).
     * timeFilter: "last_week", "last_month", "last_year" or null for no limit.
     */
    fun hitIds(terms: List<String>, timeFilter: String? = null, limit: Int = 50): List<Long> {
        val ws = uniqueWords(terms.joinToString(" "))
        if (ws.isEmpty()) return emptyList()
        val start = cutoffDate(timeFilter)
        val n = limit.coerceIn(1, 1000)

        if (h.hasFts) {
            val match = ftsQuery(terms)
            if (match != null) {
                var sql = "SELECT memories.id FROM mem_fts JOIN memories ON memories.id = mem_fts.rowid " +
                    "WHERE mem_fts MATCH ?"
                val args = mutableListOf(match)
                if (start != null) {
                    sql += " AND memories.event_date >= ?"
                    args.add(start)
                }
                sql += " ORDER BY mem_fts.rank LIMIT $n"
                try {
                    return readIds(sql, args.toTypedArray())
                } catch (e: SQLiteException) {
                    // fall through to the slower search below
                }
            }
        }

        val conds = ws.joinToString(" OR ") {
            "(lower(question) LIKE ? OR lower(answer) LIKE ? OR lower(tags) LIKE ?)"
        }
        val args = mutableListOf<String>()
        for (w in ws) {
            args.add("%$w%")
            args.add("%$w%")
            args.add("%$w%")
        }
        var sql = "SELECT id FROM memories WHERE ($conds)"
        if (start != null) {
            sql += " AND event_date >= ?"
            args.add(start)
        }
        sql += " ORDER BY id DESC LIMIT $n"
        return readIds(sql, args.toTypedArray())
    }

    /** How many memories match (at most 20). Cheap check before the heavy search. */
    fun quickCount(terms: List<String>, timeFilter: String? = null): Int =
        hitIds(terms, timeFilter, 20).size

    private fun readIds(sql: String, args: Array<String>): List<Long> {
        val out = ArrayList<Long>()
        db.rawQuery(sql, args).use { c ->
            while (c.moveToNext()) out.add(c.getLong(0))
        }
        return out
    }

    // ---------------------------------------------------------------- vectors

    /** Memories that have no vector yet (oldest first). */
    fun pending(): List<PendingRow> {
        val out = ArrayList<PendingRow>()
        db.rawQuery(
            "SELECT id, tags, question, answer FROM memories WHERE vec IS NULL ORDER BY id", null
        ).use { c ->
            while (c.moveToNext()) {
                out.add(PendingRow(c.getLong(0), c.getString(1), c.getString(2), c.getString(3)))
            }
        }
        return out
    }

    fun pendingCount(): Int {
        db.rawQuery("SELECT COUNT(*) FROM memories WHERE vec IS NULL", null).use { c ->
            return if (c.moveToFirst()) c.getInt(0) else 0
        }
    }

    /** Stores vectors (32-bit floats, same format as the Python memory.db). */
    fun setVecs(pairs: List<Pair<Long, FloatArray>>) {
        db.beginTransaction()
        try {
            for ((id, vec) in pairs) {
                db.execSQL("UPDATE memories SET vec = ? WHERE id = ?", arrayOf<Any?>(floatsToBytes(vec), id))
            }
            db.setTransactionSuccessful()
        } finally {
            db.endTransaction()
        }
    }

    /**
     * Ids of the k memories whose vectors are closest to qvec (best first).
     * Vectors of a different size (another embedder) are skipped.
     */
    fun nearest(qvec: FloatArray, timeFilter: String? = null, k: Int = MemoryConfig.TOP_K): List<Long> {
        var sql = "SELECT id, vec FROM memories WHERE vec IS NOT NULL"
        val args = ArrayList<String>()
        val start = cutoffDate(timeFilter)
        if (start != null) {
            sql += " AND event_date >= ?"
            args.add(start)
        }

        // Keep only the k best while reading: the smallest score sits at the top of the queue.
        val best = PriorityQueue<Pair<Float, Long>>(k + 1) { x, y -> x.first.compareTo(y.first) }
        db.rawQuery(sql, args.toTypedArray()).use { c ->
            while (c.moveToNext()) {
                val blob = c.getBlob(1) ?: continue
                if (blob.size != qvec.size * 4) continue
                val floats = ByteBuffer.wrap(blob).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
                var score = 0f
                for (i in qvec.indices) score += qvec[i] * floats.get(i)
                if (best.size < k) {
                    best.add(Pair(score, c.getLong(0)))
                } else if (score > best.peek()!!.first) {
                    best.poll()
                    best.add(Pair(score, c.getLong(0)))
                }
            }
        }
        return best.sortedByDescending { it.first }.map { it.second }
    }

    // ---------------------------------------------------------------- reading

    /** The memories with these ids, in the same order as ids (missing ids are skipped). */
    fun getRows(ids: List<Long>): List<MemoryRow> {
        if (ids.isEmpty()) return emptyList()
        val found = HashMap<Long, MemoryRow>()
        for (chunk in ids.chunked(500)) {
            val marks = chunk.joinToString(",") { "?" }
            db.rawQuery(
                "SELECT id, event_date, question, answer, tags FROM memories WHERE id IN ($marks)",
                chunk.map { it.toString() }.toTypedArray()
            ).use { c ->
                while (c.moveToNext()) {
                    found[c.getLong(0)] = MemoryRow(
                        c.getLong(0), c.getString(1), c.getString(2), c.getString(3), c.getString(4)
                    )
                }
            }
        }
        return ids.mapNotNull { found[it] }
    }

    /** Newest memories first, for the Memories screen. */
    fun recent(limit: Int = 50, offset: Int = 0): List<MemoryRow> {
        val out = ArrayList<MemoryRow>()
        db.rawQuery(
            "SELECT id, event_date, question, answer, tags FROM memories " +
                "ORDER BY id DESC LIMIT ${limit.coerceIn(1, 500)} OFFSET ${offset.coerceAtLeast(0)}",
            null
        ).use { c ->
            while (c.moveToNext()) {
                out.add(MemoryRow(c.getLong(0), c.getString(1), c.getString(2), c.getString(3), c.getString(4)))
            }
        }
        return out
    }
}
