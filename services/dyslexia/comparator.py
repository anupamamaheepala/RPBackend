# # services/dyslexia/comparator.py

# from difflib import ndiff

# def compare_text(reference, student):
#     diff = list(ndiff(reference, student))

#     errors = []
#     index = 0

#     for d in diff:
#         code = d[0]
#         char = d[-1]

#         if code == "-":  # missing
#             errors.append({
#                 "type": "missing",
#                 "char": char,
#                 "position": index
#             })

#         elif code == "+":  # extra
#             errors.append({
#                 "type": "extra",
#                 "char": char,
#                 "position": index
#             })

#         index += 1

#     return errors


from difflib import ndiff

def compare_text(reference, student):
    diff = list(ndiff(reference, student))

    errors = []
    index = 0

    i = 0
    while i < len(diff):
        d = diff[i]
        code = d[0]
        char = d[-1]

        # Replace case: "-x" followed by "+y"
        if code == "-" and i + 1 < len(diff) and diff[i + 1][0] == "+":
            errors.append({
                "type": "replace",
                "char": char,
                "position": index
            })
            i += 2
            index += 1
            continue

        elif code == "-":
            errors.append({
                "type": "missing",
                "char": char,
                "position": index
            })

        elif code == "+":
            errors.append({
                "type": "extra",
                "char": char,
                "position": index
            })

        index += 1
        i += 1

    return errors