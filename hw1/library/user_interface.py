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

    print('If you wish to search for a library item, write "search item".')

    print('if you wish to show all items in the library, write "show items".')

    print('if you wish to add a user to the library users, write "add user".')

    print('if you wish to show all users in the library, write "show users".')

    print('If you wish to search a user\'s checked books by name, write "user name".')

    print('If you wish to check out an item for a user, write "check out item".')

    print('If you wish to exit, write "exit".')

    command = input("What would you like to do?")

    if command not in ["add item", "search item", "show items", "add user", "show users", "user name", "check out item", "exit"]:
        print("Please follow instructions carefully!")
        continue

    if command == "add item":
        item_type = input("Please choose type (book/DVD/magazine).").lower()

        while item_type not in ["book", "dvd", "magazine"]:
            print("Item type should be only (book/DVD/magazine).")
            item_type = input("Please choose type (book/DVD/magazine).").lower()

        print("Please add the following:")
        title = input("What is the item\'s title?")
        author = input("Who is the item\'s author?")
        publication_year = int(input("What is the item\'s publication year?"))
        item_id = int(input("What is the item\'s item id?"))

        if item_type == "book":
            publisher = input("Who is the item\'s publisher?")
            number_of_pages = int(input("What is the item\'s number of pages?"))
            book = Book(title, author, publication_year, item_id, publisher, number_of_pages)
            library.add_item(book)

        if item_type == "dvd":
            director = input("Who is the item\'s director?")
            length = int(input("What is the item\'s length?"))
            rating = float(input("What is the item\'s rating?"))
            dvd = DVD(title, author, publication_year, item_id, director, length, rating)
            library.add_item(dvd)

        if item_type == "magazine":
            publisher = input("Who is the item\'s publisher?")
            issue_number = int(input("What is the item\'s issue number?"))
            magazine = Magazine(title, author, publication_year, item_id, publisher, issue_number)
            library.add_item(magazine)

        print('Item added successfully!')
        print(library)
    
