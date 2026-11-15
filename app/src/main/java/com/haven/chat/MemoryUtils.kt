package com.haven.chat

import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.time.LocalDate

/**
 * Small helper functions for the memory database.
 * Kotlin port of the helpers in store.py. Pure Kotlin (no Android), so they can be tested alone.
 */

/** Memory sizes (from config.py). */
object MemoryConfig {
    const val MAX_ANSWER_CHARS = 2000     // longest answer stored
    const val EMBED_ANSWER_CHARS = 600    // how much of the answer is embedded
    const val RESULT_ANSWER_CHARS = 600   // answer length shown to Gemma per memory
    const val TOP_K = 50                  // candidates taken from the vectors
    const val KEYWORD_EXTRA = 10          // extra candidates taken from keyword search
    const val TOP_N = 3                   // memories kept after reranking
}

private val STOP_WORDS: Set<String> = (
    "the and for are was were you your with from that this have has had can could " +
    "would should did does how what when where why who which about tell show check please " +
    "remember memory old past into out not but all any our his her its they them their there " +
    "then than also just like"
    ).split(" ").filter { it.isNotEmpty() }.toSet()

// Letters and digits of any language (Hindi, etc.), and underscore.
private val WORD_REGEX = Regex("[\\p{L}\\p{N}_]+")

/** Searchable words: lowercase, 3+ letters, no filler words. */
fun words(text: String?): List<String> {
    if (text == null) return emptyList()
    return WORD_REGEX.findAll(text.lowercase())
        .map { it.value }
        .filter { it.length > 2 && it !in STOP_WORDS }
        .toList()
}

/** The same words without repeats, in the order they first appear. */
fun uniqueWords(text: String?): List<String> = words(text).distinct()

/** At most 6 tags made from the text itself (used when Gemma gives no tags). */
fun autoTags(text: String?): List<String> = uniqueWords(text).take(6)

/**
 * Oldest allowed date (yyyy-MM-dd) for a time filter, or null for "no limit".
 * timeFilter: "last_week", "last_month", "last_year", anything else = no limit.
 */
fun cutoffDate(timeFilter: String?, today: LocalDate = LocalDate.now()): String? {
    val days = when (timeFilter) {
        "last_week" -> 7L
        "last_month" -> 31L
        "last_year" -> 366L
        else -> return null
    }
    return today.minusDays(days).toString()
}

/** Tags as one stored text: whitespace tidied, lowercase, no repeats, joined by spaces. */
fun tagText(tags: List<String>): String {
    return tags
        .map { it.trim().split(Regex("\\s+")).filter { p -> p.isNotEmpty() }.joinToString(" ").lowercase() }
        .filter { it.isNotEmpty() }
        .distinct()
        .joinToString(" ")
}

/** The text that gets embedded and reranked for one memory. */
fun embedText(tags: String, question: String, answer: String): String {
    return "$tags. $question " + answer.take(MemoryConfig.EMBED_ANSWER_CHARS)
}

/**
 * FTS5 search text: every word with a * at the end (prefix match), joined with OR.
 * Returns null when there are no usable words.
 */
fun ftsQuery(terms: List<String>): String? {
    val ws = uniqueWords(terms.joinToString(" "))
    if (ws.isEmpty()) return null
    return ws.joinToString(" OR ") { "\"$it\"*" }
}

/** Vector to bytes: 32-bit floats, little-endian (same format as the Python memory.db). */
fun floatsToBytes(v: FloatArray): ByteArray {
    val buf = ByteBuffer.allocate(v.size * 4).order(ByteOrder.LITTLE_ENDIAN)
    buf.asFloatBuffer().put(v)
    return buf.array()
}

/** Bytes back to a vector. */
fun bytesToFloats(b: ByteArray): FloatArray {
    val out = FloatArray(b.size / 4)
    ByteBuffer.wrap(b).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(out)
    return out
}

/** Dot product. Vectors are normalized, so this is the cosine similarity. */
fun dotProduct(a: FloatArray, b: FloatArray): Float {
    var sum = 0f
    val n = minOf(a.size, b.size)
    for (i in 0 until n) sum += a[i] * b[i]
    return sum
}
