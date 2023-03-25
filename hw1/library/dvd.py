from .library_item import LibraryItem

class DVD(LibraryItem):
    """A subclass of LibraryItem that represents the DVD"""

    def __init__(self, title, author, publication_year, item_id, director, length, rating):
        """Initialize the DVD"""
        super().__init__(title, author, publication_year, item_id)
        self.director = director
        self.length = length
        self.rating = rating

    def __repr__(self):
        """Return a string representation of the DVD"""
        return f'DVD(title={self.title}, author={self.author}, publication_year={self.publisher_year}, item_id={self.item_id}, director={self.director}, length={self.length}, rating={self.rating})'