from pydantic import BaseModel, EmailStr, Field
from typing import Optional
class Student(BaseModel):
    name: str = "Piyush"
    age: Optional[int] = None
    email: EmailStr
    cgpa: float = Field(gt=0, lt=10, description="CGPA must be between 0 and 10")
new_student = {'age': '20', 'email': 'abc@gmail.com', 'cgpa': 5}
student = Student(**new_student)

print(student)

student_dict = dict(student)

print(student_dict['age'])

student_json = student.model_dump_json()
print(student_json)