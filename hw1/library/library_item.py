class LibraryItem:
    """A class representing a library item"""

    def __init__(self, title, author, publication_year, item_id):
        """Initialize the library item"""
        self.title = title
        self.author = author
        self.publisher_year = publication_year
        self.item_id = item_id

    def __repr__(self):
        """Return a string representation of the library item"""
        return f'LibraryItem(title={self.title}, author={self.author}, publication_year={self.publisher_year}, item_id={self.item_id})'
