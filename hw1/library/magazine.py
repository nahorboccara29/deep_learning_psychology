from .library_item import LibraryItem


class Magazine(LibraryItem):
    """A subclass of LibraryItem that represents the magazine"""

    def __init__(self, title: str, author: str, publication_year: int, item_id: int, publisher: str, issue_number: int):
        """Initialize the magazine"""
        super().__init__(title, author, publication_year, item_id)
        self.publisher = publisher
        self.issue_number = issue_number

    def __repr__(self):
        """Return a string representation of the magazine"""
        return f'Magazine(title={self.title}, author={self.author}, publication_year={self.publication_year}, item_id={self.item_id}, publisher={self.publisher}, issue_number={self.issue_number})'
