from .library import Library
from .book import Book
from .dvd import DVD
from .magazine import Magazine
from .library_item import LibraryItem
from .library_user import LibraryUser


library = Library()

command = ''

while command != "exit":

    print("Hello, welcome to the library!")
    print('If you wish to add an item to the library, write "add item".')
    print('If you wish to search for a library item, write "item name".')
    print('if you wish to show all items in the library, write "show items".')
    print('if you wish to add a user to the library users, write "add user".')
    print('if you wish to show all users in the library, write "show users".')
    print('If you wish to search a user\'s checked books by name, write "user name".')
    print('If you wish to check out an item for a user, write "check out item".')
    print('If you wish to return an item for a user, write "return item".')
    print('If you wish to exit, write "exit".')
    print()

    command = input("What would you like to do? ")

    if command not in ["add item", "item name", "show items", "add user", "show users", "user name", "check out item", "return item", "exit"]:
        print("Please follow instructions carefully!")
        continue

    if command == "add item":
        item_type = input("Please choose type (book/DVD/magazine). ").lower()

        while item_type not in ["book", "dvd", "magazine"]:
            print("Item type should be only (book/DVD/magazine).")
            item_type = input(
                "Please choose type (book/DVD/magazine). ").lower()

        print("Please add the following:")
        title = input("What is the item\'s title? ")
        author = input("Who is the item\'s author? ")
        publication_year = int(input("What is the item\'s publication year? "))
        item_id = int(input("What is the item\'s item id? "))

        if item_type == "book":
            publisher = input("Who is the item\'s publisher? ")
            number_of_pages = int(
                input("What is the item\'s number of pages? "))
            book = Book(title, author, publication_year,
                        item_id, publisher, number_of_pages)
            library.add_item(book)

        if item_type == "dvd":
            director = input("Who is the item\'s director? ")
            length = int(input("What is the item\'s length? "))
            rating = float(input("What is the item\'s rating? "))
            dvd = DVD(title, author, publication_year,
                      item_id, director, length, rating)
            library.add_item(dvd)

        if item_type == "magazine":
            publisher = input("Who is the item\'s publisher? ")
            issue_number = int(input("What is the item\'s issue number? "))
            magazine = Magazine(title, author, publication_year,
                                item_id, publisher, issue_number)
            library.add_item(magazine)

        print('Item added successfully!')

    if command == "item name":
        print("Please add the following:")
        name = input("What is the item\'s name? ")
        try:
            item = library.get_item_by_name(name)
            print(item)
        except ValueError as e:
            print(f'Error: {e}')
            continue

    if command == "show items":
        library.get_items_report()

    if command == "add user":
        print("Please add the following:")
        user_id = int(input("What is the users\'s user id? "))
        name = input("What is the users\'s name? ")
        library_user = LibraryUser(user_id, name)
        library.add_user(library_user)
        print('User added successfully!')

    if command == "show users":
        library.get_users_report()

    if command == "user name":
        print("Please add the following:")
        name = input("What is the user\'s name? ")
        try:
            user = library.get_user_by_name(name)
            print(user)
        except ValueError as e:
            print(f'Error: {e}')
            continue

    if command == "check out item":
        print("Please add the following:")
        item_id = int(input("What is the item\'s item id? "))
        user_id = int(input("What is the users\'s user id? "))
        try:
            library.check_out_item(user_id, item_id)
        except ValueError as e:
            print(f'Error: {e}')
            continue

    if command == "return item":
        print("Please add the following:")
        item_id = int(input("What is the item\'s item id? "))
        user_id = int(input("What is the users\'s user id? "))
        try:
            library.return_item(user_id, item_id)
        except ValueError as e:
            print(f'Error: {e}')
            continue

    print()
