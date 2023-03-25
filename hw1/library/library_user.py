from typing import List

from .library_item import LibraryItem


class LibraryUser:
    """A class representing a library user"""

    def __init__(self, user_id: int, name: str, items_checked_out: List[LibraryItem]):
        """Initialize the library user"""
        self.user_id = user_id
        self.name = name
        self.items_checked_out = items_checked_out

    def check_out_item(self, item: LibraryItem):
        """Check out a library item"""
        self.items_checked_out.append(item)

    def return_item(self, item_id: int) -> LibraryItem:
        """Return a library item"""
        for i, item in enumerate(self.items_checked_out):
            if item_id == item.item_id:
                return self.items_checked_out.pop(i)
        raise ValueError("Item not found")

    def __repr__(self):
        """Return a string representation of the library user"""
        return f'LibraryUser(user_id={self.user_id}, name={self.name}, items_checked_out={self.items_checked_out})'
