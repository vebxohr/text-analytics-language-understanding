
words = ['she', 'sells', 'sea', 'shells', 'by', 'the', 'sea', 'shore']


# task a)
print("Task a):")
[print(word) for word in words if word[:2] == 'sh']


# task b)
print("\nTask b):")
[print(word) for word in words if len(word) > 4]
