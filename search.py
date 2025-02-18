import lancedb

def main():
    # Connect to the database
    uri = "data/lancedb"
    db = lancedb.connect(uri)

    # Load the table
    table = db.open_table("docling")

    # Define a GDPR-specific query
    query = "What are the key principles of GDPR data protection?"

    print("Searching for GDPR-related content...")
    print("Query:", query)

    # Search the table for the GDPR-specific query, limiting to 5 results
    result = table.search(query=query).limit(5)
    df = result.to_pandas()

    # Display a summary of the search results
    print("\nSearch Results (first 5 rows):")
    print(df.head())
    print(f"\nTotal rows returned: {df.shape[0]}, Table shape: {df.shape}")

if __name__ == "__main__":
    main()
