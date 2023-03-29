from .library_item import LibraryItem


class DVD(LibraryItem):
    """A subclass of LibraryItem that represents the DVD"""

    def __init__(self, title: str, author: str, publication_year: int, item_id: int, director: str, length: int, rating: float):
        """Initialize the DVD"""
        super().__init__(title, author, publication_year, item_id)
        self.director = director
        self.length = length
        self.rating = rating

    def __repr__(self):
        """Return a string representation of the DVD"""
        return f'DVD(title={self.title}, author={self.author}, publication_year={self.publication_year}, item_id={self.item_id}, director={self.director}, length={self.length}, rating={self.rating})'
