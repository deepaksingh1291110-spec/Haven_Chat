package com.haven.chat

import android.database.sqlite.SQLiteDatabase
import android.database.sqlite.SQLiteException
import java.io.File

/**
 * The one database file of the app.
 *
 * Tables:
 *   memories  what Gemma chose to remember (same layout as the Python memory.db)
 *   mem_fts   keyword search index for memories (FTS5, if this phone supports it)
 *   chats     the chat list
 *   messages  the messages of each chat
 *
 * The file lives in shared storage (survives reinstalling the app), so the app needs
 * "All files access". Use MemoryStore and ChatStore to read and write; they get their
 * database from here.
 */
class HavenDb private constructor(
    val db: SQLiteDatabase,
    val path: String,
    /** True when the FTS5 keyword index is available; otherwise slower word matching is used. */
    val hasFts: Boolean
) {

    /** Number of rows in a table (table name comes from our own code, never from the user). */
    fun count(table: String): Int {
        db.rawQuery("SELECT COUNT(*) FROM $table", null).use { c ->
            return if (c.moveToFirst()) c.getInt(0) else 0
        }
    }

    /** One line about the database, for the self-test screen. */
    fun info(): String {
        return "database " + path + " | keyword index (FTS5): " + (if (hasFts) "yes" else "no (slower fallback)") +
            " | memories " + count("memories") + ", chats " + count("chats") + ", messages " + count("messages")
    }

    fun close() {
        db.close()
    }

    companion object {
        const val DEFAULT_PATH = "/storage/emulated/0/HavenData/memory.db"
        private const val SCHEMA_VERSION = 1

        private val TABLES = listOf(
            """CREATE TABLE IF NOT EXISTS memories(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                event_date TEXT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                tags TEXT NOT NULL,
                vec BLOB)""",
            """CREATE TABLE IF NOT EXISTS chats(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL)""",
            """CREATE TABLE IF NOT EXISTS messages(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                thinking TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(chat_id) REFERENCES chats(id) ON DELETE CASCADE)""",
            "CREATE INDEX IF NOT EXISTS idx_messages_chat ON messages(chat_id, id)",
            "CREATE INDEX IF NOT EXISTS idx_chats_updated ON chats(updated_at)"
        )

        private const val FTS_TABLE =
            "CREATE VIRTUAL TABLE IF NOT EXISTS mem_fts USING fts5(" +
                "question, answer, tags, tokenize='porter unicode61')"

        /**
         * Opens the database file (creates it, and its folder, if needed).
         * Throws an exception with a clear message if the file cannot be opened,
         * for example when "All files access" is not allowed yet.
         */
        fun open(path: String = DEFAULT_PATH): HavenDb {
            val folder = File(path).parentFile
            if (folder != null && !folder.isDirectory && !folder.mkdirs()) {
                throw IllegalStateException("Cannot create the folder " + folder.path + " (is All files access allowed?)")
            }
            val db = SQLiteDatabase.openOrCreateDatabase(path, null)
            try {
                db.setForeignKeyConstraintsEnabled(true)
                for (sql in TABLES) db.execSQL(sql)

                var fts = true
                try {
                    db.execSQL(FTS_TABLE)
                } catch (e: SQLiteException) {
                    fts = false
                }

                if (fts) fillFtsIfEmpty(db)
                if (db.version < SCHEMA_VERSION) db.version = SCHEMA_VERSION
                return HavenDb(db, path, fts)
            } catch (e: Exception) {
                db.close()
                throw e
            }
        }

        // A memory.db copied in from the Python system may have no keyword index yet: build it.
        private fun fillFtsIfEmpty(db: SQLiteDatabase) {
            val indexed = db.rawQuery("SELECT COUNT(*) FROM mem_fts", null).use { c ->
                if (c.moveToFirst()) c.getInt(0) else 0
            }
            val saved = db.rawQuery("SELECT COUNT(*) FROM memories", null).use { c ->
                if (c.moveToFirst()) c.getInt(0) else 0
            }
            if (indexed == 0 && saved > 0) {
                db.execSQL(
                    "INSERT INTO mem_fts(rowid, question, answer, tags) " +
                        "SELECT id, question, answer, tags FROM memories"
                )
            }
        }
    }
}
