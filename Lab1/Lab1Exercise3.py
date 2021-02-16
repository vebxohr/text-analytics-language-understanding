from nltk.tokenize import word_tokenize

# Task a
vowels = set('aeiou')  # English vowels


def pig_latin_word(word):
    letters_to_move = []
    for char in word:
        if char not in vowels:
            letters_to_move.append(char)
        else:
            break

    return word[len(letters_to_move):] + ''.join(letters_to_move) + 'ay'


print(pig_latin_word('string'))


# Task b


def pig_latin_text(text):
    tokens = word_tokenize(text)
    final_text = ''
    for word in tokens:
        if word.isalpha():
            result = pig_latin_word(word)
        else:
            result = word
            final_text = final_text[:-1]  # to place punctuation marks correctly

        final_text += result + ' '

    return final_text


print(pig_latin_text('Happy birthday, John'))


# Task c

# def pig_latin_decode_word(word):
#     word = word[:-2]  # Remove 'ay'
#     letters_to_move = []
#     for i in range(len(word) - 1, 0, -1):
#         char = word[i]
#         if char not in vowels:
#             letters_to_move.insert(0, char)
#         else:
#             break
#
#     return ''.join(letters_to_move) + word[:len(word)-len(letters_to_move)]
#
# print(pig_latin_decode_word("ingstray"))
#
# def pig_latin_decode_text(text):
#     tokens = word_tokenize(text)
#     final_text = ''
#
#     for word in tokens:
#         if word.isalpha():
#             pass
