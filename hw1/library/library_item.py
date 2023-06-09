class LibraryItem:
    """A class representing a library item"""

    def __init__(self, title: str, author: str, publication_year: int, item_id: int):
        """Initialize the library item"""
        self.title = title
        self.author = author
        self.publication_year = publication_year
        self.item_id = item_id

    def __repr__(self):
        """Return a string representation of the library item"""
        return f'LibraryItem(title={self.title}, author={self.author}, publication_year={self.publication_year}, item_id={self.item_id})'
