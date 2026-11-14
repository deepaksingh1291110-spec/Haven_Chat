package com.haven.chat

/** Receives the answer piece by piece. Return false to stop generating. */
interface TokenListener {
    fun onToken(bytes: ByteArray): Boolean
}

/**
 * Kotlin side of the native engine (haven_engine.cpp).
 * Create ONE instance and call everything from ONE background thread,
 * except stop(), which may be called from any thread.
 */
class LlamaBridge {

    /** Call once at app start with applicationInfo.nativeLibraryDir. */
    external fun init(nativeLibDir: String)

    /** Returns 0 = ok, 1 = model file failed to load, 2 = context failed to create. */
    external fun loadModel(path: String, nCtx: Int, nThreads: Int): Int

    /** Change the CPU thread count without reloading the model. */
    external fun setThreads(nThreads: Int)

    external fun systemInfo(): String

    /** Start a new conversation. systemPrompt is UTF-8 bytes (may be empty). */
    external fun resetChat(systemPrompt: ByteArray)

    /** Sends one user message (UTF-8 bytes). Returns a stats line, or text starting with "ERROR:". */
    external fun generate(
        userText: ByteArray,
        maxNew: Int,
        thinking: Boolean,
        listener: TokenListener
    ): String

    external fun stop()

    external fun unload()

    /** Last warnings and errors printed by llama.cpp. */
    external fun lastLog(): String

    // ---------------------------------------------------------------- search models
    // Only ONE search model is loaded at a time, and only while the main model is
    // unloaded (RAM). Order: unload() -> loadAux(embedder) -> embed(...) -> unloadAux()
    //                        -> loadAux(reranker) -> rerankScore(...) -> unloadAux()

    /**
     * Loads a search model. kind: KIND_EMBEDDER or KIND_RERANKER.
     * Returns 0 = ok, 1 = model file failed to load, 2 = context failed to create,
     * 3 = unknown kind, 4 = the main model is still loaded (call unload() first).
     */
    external fun loadAux(path: String, kind: Int, nThreads: Int): Int

    /** One line of facts about the loaded search model (vector size, pooling type). */
    external fun auxInfo(): String

    /** Embeds one text (UTF-8 bytes). Returns a normalized vector, or null on error. */
    external fun embed(text: ByteArray): FloatArray?

    /** Relevance score of a document for a query (UTF-8 bytes). Higher = better. -1e6 = error. */
    external fun rerankScore(query: ByteArray, doc: ByteArray): Float

    external fun unloadAux()

    companion object {
        const val KIND_EMBEDDER = 1
        const val KIND_RERANKER = 2

        init {
            System.loadLibrary("haven-engine")
        }
    }
}
