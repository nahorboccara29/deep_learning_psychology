from typing import List

from .library_item import LibraryItem
from .library_user import LibraryUser


class LibraryUser:
    """A class representing a library user"""

    def __init__(self, items: List[LibraryItem] = None, users: List[LibraryUser] = None):
        """Initialize the library user"""
        self.items = items if items is not None else []
        self.users = users if users is not None else []

    def add_item(self, item: LibraryItem):
        """Add an item to the library items"""
        self.items.append(item)

    def remove_item(self, item_id: int):
        """Remove an item from the library items"""
        item = self._get_item(item_id)
        self.items.remove(item)

    def add_user(self, user: LibraryUser):
        """Add a user to the library users"""
        self.users.append(user)

    def remove_user(self, user_id: int):
        """Remove a user from the library users"""
        user = self._get_user(user_id)
        self.users.remove(user)

    def check_out_item(self, user_id: int, item_id: int):
        """Check out an item to a user and remove it from the library items"""
        user = self._get_user(user_id)
        item = self._get_item(item_id)
        user.check_out_item(item)
        self.remove_item(item_id)

    def return_item(self, user_id: int, item_id: int):
        """Return an item from a user and add it to the library items"""
        user = self._get_user(user_id)
        item = self._get_item(item_id)
        user.return_item(item_id)
        self.add_item(item)

    def _get_user(self, user_id: int) -> LibraryUser:
        """Get a user from the library users by its user id"""
        for user in self.users:
            if user_id == user.user_id:
                return user
        raise ValueError("User not found")

    def _get_item(self, item_id: int) -> LibraryItem:
        """Get an item from the library items by its item id"""
        for item in self.items:
            if item_id == item.item_id:
                return item
        raise ValueError("Item not found")

    def __repr__(self):
        """Return a string representation of the library user"""
        return f'LibraryUser(items={self.items}, users={self.users})'

    def get_items_report(self):
        print("Items currently in Library:")
        print('\n\n'.join([str(item) for item in self.items]))

    def get_users_report(self):
        print("Users currently in Library:")
        print('\n\n'.join([str(user) for user in self.users]))
