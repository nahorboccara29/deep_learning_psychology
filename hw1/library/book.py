from .library_item import LibraryItem

class Book(LibraryItem):
    """A subclass of LibraryItem that represents the book"""

    def __init__(self, title, author, publication_year, item_id, publisher, number_of_pages):
        """Initialize the book"""
        super().__init__(title, author, publication_year, item_id)
        self.publisher = publisher
        self.number_of_pages = number_of_pages

    def __repr__(self):
        """Return a string representation of the book"""
        return f'Book(title={self.title}, author={self.author}, publication_year={self.publisher_year}, item_id={self.item_id}, publisher={self.publisher}, number_of_pages={self.number_of_pages})'

