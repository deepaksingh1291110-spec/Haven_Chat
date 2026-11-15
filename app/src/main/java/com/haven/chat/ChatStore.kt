package com.haven.chat

import java.time.LocalDateTime
import java.time.format.DateTimeFormatter

/** One entry of the chat list. */
data class ChatRow(
    val id: Long,
    val title: String,
    val createdAt: String,
    val updatedAt: String
)

/** One message of a chat. role is "user" or "assistant". thinking is Gemma's thinking text, if any. */
data class MessageRow(
    val id: Long,
    val chatId: Long,
    val role: String,
    val content: String,
    val thinking: String?,
    val createdAt: String
)

private val CHAT_TIME_FORMAT = DateTimeFormatter.ofPattern("yyyy-MM-dd'T'HH:mm:ss")

private fun chatNow(): String = LocalDateTime.now().format(CHAT_TIME_FORMAT)

/**
 * Chat history, like Claude or Gemini: a list of chats, and the full messages of each chat.
 * This is separate from long-term memory (MemoryStore): chat history is the whole
 * conversation, memories are only what Gemma chose to remember.
 * Call from a background thread.
 */
class ChatStore(private val h: HavenDb) {

    private val db get() = h.db

    // ---------------------------------------------------------------- chats

    /** Creates an empty chat and returns its id. */
    fun createChat(title: String = DEFAULT_TITLE): Long {
        val now = chatNow()
        db.execSQL(
            "INSERT INTO chats(title, created_at, updated_at) VALUES (?,?,?)",
            arrayOf<Any?>(title, now, now)
        )
        db.rawQuery("SELECT last_insert_rowid()", null).use { c ->
            return if (c.moveToFirst()) c.getLong(0) else -1L
        }
    }

    /** All chats, most recently used first. */
    fun listChats(limit: Int = 200): List<ChatRow> {
        val out = ArrayList<ChatRow>()
        db.rawQuery(
            "SELECT id, title, created_at, updated_at FROM chats " +
                "ORDER BY updated_at DESC, id DESC LIMIT ${limit.coerceIn(1, 1000)}",
            null
        ).use { c ->
            while (c.moveToNext()) {
                out.add(ChatRow(c.getLong(0), c.getString(1), c.getString(2), c.getString(3)))
            }
        }
        return out
    }

    fun getChat(chatId: Long): ChatRow? {
        db.rawQuery(
            "SELECT id, title, created_at, updated_at FROM chats WHERE id = ?",
            arrayOf(chatId.toString())
        ).use { c ->
            return if (c.moveToFirst()) {
                ChatRow(c.getLong(0), c.getString(1), c.getString(2), c.getString(3))
            } else null
        }
    }

    fun renameChat(chatId: Long, title: String) {
        val t = titleFrom(title)
        db.execSQL("UPDATE chats SET title = ? WHERE id = ?", arrayOf<Any?>(t, chatId))
    }

    /** Deletes a chat together with all its messages. */
    fun deleteChat(chatId: Long) {
        db.beginTransaction()
        try {
            db.execSQL("DELETE FROM messages WHERE chat_id = ?", arrayOf<Any?>(chatId))
            db.execSQL("DELETE FROM chats WHERE id = ?", arrayOf<Any?>(chatId))
            db.setTransactionSuccessful()
        } finally {
            db.endTransaction()
        }
    }

    fun chatCount(): Int = h.count("chats")

    // ---------------------------------------------------------------- messages

    /**
     * Adds a message to a chat and returns its id.
     * The chat moves to the top of the list. If the chat still has the default title,
     * the first user message becomes its title (no extra Gemma call needed).
     */
    fun addMessage(chatId: Long, role: String, content: String, thinking: String? = null): Long {
        val now = chatNow()
        var newId = -1L
        db.beginTransaction()
        try {
            db.execSQL(
                "INSERT INTO messages(chat_id, role, content, thinking, created_at) VALUES (?,?,?,?,?)",
                arrayOf<Any?>(chatId, role, content, thinking, now)
            )
            db.rawQuery("SELECT last_insert_rowid()", null).use { c ->
                if (c.moveToFirst()) newId = c.getLong(0)
            }
            db.execSQL("UPDATE chats SET updated_at = ? WHERE id = ?", arrayOf<Any?>(now, chatId))
            if (role == "user") {
                db.execSQL(
                    "UPDATE chats SET title = ? WHERE id = ? AND title = ?",
                    arrayOf<Any?>(titleFrom(content), chatId, DEFAULT_TITLE)
                )
            }
            db.setTransactionSuccessful()
        } finally {
            db.endTransaction()
        }
        return newId
    }

    /** All messages of a chat, oldest first. */
    fun messages(chatId: Long): List<MessageRow> {
        return readMessages(
            "SELECT id, chat_id, role, content, thinking, created_at FROM messages " +
                "WHERE chat_id = ? ORDER BY id",
            arrayOf(chatId.toString())
        )
    }

    /** The last n messages of a chat, oldest first (to continue an old chat with limited context). */
    fun lastMessages(chatId: Long, n: Int): List<MessageRow> {
        return readMessages(
            "SELECT id, chat_id, role, content, thinking, created_at FROM messages " +
                "WHERE chat_id = ? ORDER BY id DESC LIMIT ${n.coerceIn(1, 1000)}",
            arrayOf(chatId.toString())
        ).reversed()
    }

    fun messageCount(chatId: Long): Int {
        db.rawQuery("SELECT COUNT(*) FROM messages WHERE chat_id = ?", arrayOf(chatId.toString())).use { c ->
            return if (c.moveToFirst()) c.getInt(0) else 0
        }
    }

    private fun readMessages(sql: String, args: Array<String>): List<MessageRow> {
        val out = ArrayList<MessageRow>()
        db.rawQuery(sql, args).use { c ->
            while (c.moveToNext()) {
                out.add(
                    MessageRow(
                        c.getLong(0), c.getLong(1), c.getString(2), c.getString(3),
                        if (c.isNull(4)) null else c.getString(4), c.getString(5)
                    )
                )
            }
        }
        return out
    }

    companion object {
        const val DEFAULT_TITLE = "New chat"
        private const val MAX_TITLE = 40

        /** A chat title from a text: first non-empty line, tidy spaces, at most 40 characters. */
        fun titleFrom(text: String): String {
            val firstLine = text.lineSequence().map { it.trim() }.firstOrNull { it.isNotEmpty() } ?: return DEFAULT_TITLE
            val tidy = firstLine.split(Regex("\\s+")).joinToString(" ")
            return if (tidy.length <= MAX_TITLE) tidy else tidy.take(MAX_TITLE - 1).trimEnd() + "\u2026"
        }
    }
}
