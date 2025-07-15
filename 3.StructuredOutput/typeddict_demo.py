from typing import TypedDict

class Person(TypedDict):
    name: str
    age: int
    is_student: bool

new_person: Person = {'name':'Piyush', 'age': 20, 'is_student': True}

print(new_person)

old_person: Person = {'name': 'Piyush', 'age': 20, 'is_student': 'True',}
# This will not raise a type error even when 'is_student' should be a bool, not a str