import sqlite3
import sqlite_vec
import sqlite_rembed


class Memory:
    def write_memory(self, note):
        pass

    def query_memory(self, query):
        pass


class SQLiteVecMemory:
    def __init__(self, filename):
        self.db = sqlite3.connect("memory.db")
        self.db.enable_load_extension(True)
        sqlite_vec.load(self.db)
        sqlite_rembed.load(self.db)
        self.db.enable_load_extension(False)
        self.configure()
        self.migrate()

    def configure(self):
        self.db.execute("""
        INSERT INTO temp.rembed_clients(name, options) VALUES ('text-embedding-3-small', 'openai');
        """)

    def migrate(self):
        self.db.execute("""
        CREATE TABLE IF NOT EXISTS memory(
            rowid INTEGER PRIMARY KEY AUTOINCREMENT,
            note TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        self.db.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS vec_memory USING vec0(
            embeddings float[1536]
        );
        """)
        self.db.commit()

    def get_memory_by_note(self, note):
        return self.db.execute(
            "SELECT * FROM memory WHERE note = ?",
            (note,),
        ).fetchone()

    def write_memory(self, note):
        if self.get_memory_by_note(note):
            # already exists
            return
        self.db.execute(
            "INSERT INTO memory (note) VALUES (?)",
            (note,),
        )
        self.db.execute(
            "INSERT INTO vec_memory(rowid, embeddings) VALUES (last_insert_rowid(), rembed('text-embedding-3-small', ?))",
            (note,),
        )
        self.db.commit()

    def query_memory(self, query):
        return self.db.execute(
            """
        WITH MATCHES AS (
          SELECT
            rowid,
            distance
          FROM vec_memory
          WHERE embeddings MATCH rembed('text-embedding-3-small', ?)
          ORDER BY distance
          LIMIT 3
        )
        SELECT
          memory.rowid,
          memory.note,
          memory.created_at,
          matches.distance
        FROM matches
        LEFT JOIN memory ON memory.rowid = matches.rowid;
        """,
            (query,),
        )


if __name__ == "__main__":
    db = SQLiteVecMemory("memory.db")
    db.write_memory("Hello, world!")
    db.write_memory("My cat is brown")
    db.write_memory("My old cat is named Nim")
    db.write_memory("I took a class about bread")
    db.write_memory("My name is Mathieu")

    questions = [
        "What is my name ?",
        "How many cats do I have ?",
        "What did i learn on cooking recently ?"
    ]
    for question in questions:
        print("- Q:", question)
        for row in db.query_memory(question):
            print(row)
        print()
