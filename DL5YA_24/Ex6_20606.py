
def secret(s1, s2, key):
    coded = ""
    for x in s1: 
        coded += chr(ord(x)+key)
        key += 1
    return coded == s2

def secretRec(s1, s2, key):
    if (s1 == "" and s2 != "") or (s1 != "" and s2 == ""):
        return False
    if s1 == "" and s2 == "": return True
    return  s1[0] == chr(ord(s2[0])-key) and secretRec(s1[1:], s2[1:], key+1)


s1 = "abc"
s2 = "ceg"
key = 2
print(secretRec(s1,s2,key))
    