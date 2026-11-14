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

    companion object {
        init {
            System.loadLibrary("haven-engine")
        }
    }
}
