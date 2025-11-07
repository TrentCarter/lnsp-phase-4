import psycopg2

class DBConnection:
    def __init__(self, dbname, user, password, host='localhost', port=5432):
        self.connection = None
        self.dbname = dbname
        self.user = user
        self.password = password
        self.host = host
        self.port = port

    def connect(self):
        try:
            self.connection = psycopg2.connect(
                dbname=self.dbname,
                user=self.user,
                password=self.password,
                host=self.host,
                port=self.port
            )
            print("Connection successful")
        except Exception as e:
            print(f"Error connecting to database: {e}")
            raise

    def close(self):
        if self.connection:
            self.connection.close()
            print("Connection closed")

# Example usage
if __name__ == "__main__":
    db = DBConnection(dbname="your_db", user="your_user", password="your_password")
    db.connect()
    # Perform database operations here
    db.close()
