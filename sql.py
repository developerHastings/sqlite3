import sqlite3

#Connects to a database file (creates it if it doesn't exist)
conn = sqlite3.connect('school.db')
cursor = conn.cursor()

# Create the students table if it does not exist
# The table will have columns for StudentID, FirstName, LastName, Age, and Grade
cursor.execute('''
CREATE TABLE IF NOT EXISTS students (
               StudentID INTEGER PRIMARY KEY AUTOINCREMENT,
                FirstName TEXT NOT NULL,
                LastName TEXT NOT NULL,
                Age INTEGER NOT NULL,
                Grade TEXT NOT NULL
               )
''')
conn.commit()

# Insert sample data into the students table
cursor.execute("INSERT INTO students (FirstName, LastName, Age, Grade) VALUES ('John', 'Doe', 15, '10th')")
cursor.execute("INSERT INTO students (FirstName, LastName, Age, Grade) VALUES ('Jane', 'Smith', 14, '9th')")
cursor.execute("INSERT INTO students (FirstName, LastName, Age, Grade) VALUES ('Emily', 'Johnson', 16, '11th')")
cursor.execute("INSERT INTO students (FirstName, LastName, Age, Grade) VALUES ('Michael', 'Brown', 17, '12th')")
conn.commit()

# Update transaction with error handling
try:
    #Attempt to update Jane's Grade
    cursor.execute("UPDATE students SET Grade = '10th' WHERE FirstName = 'Jane' AND LastName = 'Smith'")
    # Commit the changes if successful
    conn.commit()
    print("Update successful.")
except Exception as e:
    #If an error occurs, rollback the transaction
    print("An error occurred:", e)
    conn.rollback()
    print("Update rolled back.")

#Query to select all students from the table
cursor.execute("SELECT * FROM students")
row = cursor.fetchone()
while row is not None:
    print(row) #process the row (e.g print it)
    row = cursor.fetchone() #get the next row

# Close the connection
conn.close()

