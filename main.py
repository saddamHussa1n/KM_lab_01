import sys

def write_to_file(file, string1):
    f = open(file, mode='w')
    f.write(string1)
    f.close()


def read_file(file):
    f = open(file, mode='r')
    message = ''
    for ch in f.read():
        if ch != '\n':
            message += ch
    if not message:
        print("НЕТ СООБЩЕНИЯ В ФАЙЛЕ in.txt")
        sys.exit()
    f.close()
    return message


def get_alphabet(file):
    f = open(file, mode='r')
    alphabet = ''
    for ch in f.read():
        if ch != '\n':
            alphabet += ch
    if not alphabet:
        alphabet += 'АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ '
    f.close()
    return alphabet


ALPHABET = get_alphabet("alphabet.txt")
len_of_alphabet = len(ALPHABET)


def get_key_cesar(file, array):
    f = open(file, mode='r')
    letter = ''
    for ch in f.read():
        if ch != '\n':
            letter += ch
    key = array.index(letter)
    if not letter:
        print("НЕТ КЛЮЧА В ФАЙЛЕ key.txt")
        sys.exit()
    f.close()
    return key


def get_key_affine(file, array):
    f = open(file, mode='r')
    letter1 = f.read(1)
    letter2 = f.read(1)
    key1 = array.index(letter1)
    key2 = array.index(letter2)
    if not key1 or not key2:
        print("НЕТ КЛЮЧА В ФАЙЛЕ key.txt")
        sys.exit()
    f.close()
    return key1, key2


def get_key_substitution(file):
    f = open(file, mode='r')
    key = ''
    for ch in f.read():
        if ch != '\n':
            key += ch
    if not key:
        print("НЕТ КЛЮЧА В ФАЙЛЕ key.txt")
        sys.exit()
    if len(key) != len_of_alphabet:
        print("НЕПРАВИЛЬНЫЙ КЛЮЧ")
        sys.exit()
    f.close()
    return key


def get_key_columnar(file):
    f = open(file, mode='r')
    key = ''
    for ch in f.read():
        if ch != '\n':
            key += ch
    if not key:
        print("НЕТ КЛЮЧА В ФАЙЛЕ key.txt")
        sys.exit()
    if len(key) > len_of_alphabet:
        print("НЕПРАВИЛЬНЫЙ КЛЮЧ")
        sys.exit()
    f.close()
    return key


def get_key_vigenere(file):
    f = open(file, mode='r')
    key = ''
    for ch in f.read():
        if ch != '\n':
            key += ch
    if not key:
        print("НЕТ КЛЮЧА В ФАЙЛЕ key.txt")
        sys.exit()
    f.close()
    return key


# Build shifted alphabet
def offset(char, offset):
    return ALPHABET[(ALPHABET.index(char) + offset) % len_of_alphabet]


class Caesar:
    @staticmethod
    def encrypt(message, key):
        return ''.join(map(offset, list(message), [key, ] * len(message)))

    @staticmethod
    def decrypt(ciphertext, key):
        return ''.join(map(offset, list(ciphertext), [len_of_alphabet - key, ] * len(ciphertext)))


class Vigenere:
    @staticmethod
    def encrypt(message, key):
        return ''.join(
            map(offset, message, list(map(lambda x: ALPHABET.index(x), key)) * (len(message) // len(key) + 1)))

    @staticmethod
    def decrypt(ciphertext, key):
        return ''.join(map(offset, ciphertext,
                           list(map(lambda x: len_of_alphabet - ALPHABET.index(x), key)) * (
                                   len(ciphertext) // len(key) + 1)))


class Substitution:
    @staticmethod
    def encrypt(message, key):
        cipher_alph = Substitution.buildAlphabet(key)
        return ''.join(cipher_alph[ALPHABET.index(ch)] for ch in message)

    # Built substitution alphabet by key
    @staticmethod
    def buildAlphabet(key):
        offseted_alph = ''.join(map(offset, list(ALPHABET), [ALPHABET.index(key[-1]) + 1, ] * len(ALPHABET)))
        return (key + ''.join([ch for ch in offseted_alph if not (ch in key)]))

    @staticmethod
    def decrypt(ciphertex, key):
        cipher_alph = Substitution.buildAlphabet(key)
        return ''.join(ALPHABET[cipher_alph.index(ch)] for ch in ciphertex)


class Affine:
    @staticmethod
    def modReverse(a, b):
        r, s, t = [min(a, b), max(a, b)], [1, 0], [0, 1]
        while r[-1] != 1:
            q = r[-2] // r[-1]
            r.append(r[-2] - q * r[-1])
            s.append(s[-2] - q * s[-1])
            t.append(t[-2] - q * t[-1])
        return (s[-1] % r[1])

    # key should be the tuple
    @staticmethod
    def encrypt(message, key):
        return ''.join(ALPHABET[(ALPHABET.index(ch) * key[0] + key[1]) % len_of_alphabet] for ch in message)

    # key should be the tuple
    @staticmethod
    def decrypt(ciphertext, key):
        try:
            return ''.join(
                ALPHABET[Affine.modReverse(key[0], len_of_alphabet) * (ALPHABET.index(ch) - key[1]) % len_of_alphabet]
                for ch in
                ciphertext)
        except ZeroDivisionError:
            pass


class ColumnarTransposition:
    @staticmethod
    def encrypt(message, key):
        message = message + 'x' * ((0 - len(message) % len(key)) % len(key))
        res = ''.join([message[k] for i in ColumnarTransposition.transformkey(key) for k in range(len(message)) if
                       k % len(key) == i])
        return res

    @staticmethod
    def decrypt(ciphertext, key):
        return ''.join(
            [ciphertext[ColumnarTransposition.transformkey(key).index(i) * len(ciphertext) // len(key) + k] for k in
             range(len(ciphertext) // len(key)) for i in range(len(key))])

    @staticmethod
    def transformkey(key):
        return [i[0] for i in sorted([i for i in enumerate(key)], key=lambda x: x[1])]


if __name__ == '__main__':
    which_cipher = int(input(
        '1 - Шифр сдвига\n2 - Афинный шифр\n3 - Шифр простой замены\n4 - Шифр Хилла\n5 - Шифр перестановки\n6 - Шифр Виженера\n'))

    what_to_do = int(input("1 - Зашифровать\n2 - Расшифровать\n"))

    if which_cipher == 1 and what_to_do == 1:
        test = read_file('in.txt')
        c = Caesar.encrypt(test, get_key_cesar('key.txt', ALPHABET))
        print(c)
    elif which_cipher == 1 and what_to_do == 2:
        test = read_file('in.txt')
        c = Caesar.decrypt(test, get_key_cesar('key.txt', ALPHABET))
        print(c)
    elif which_cipher == 2 and what_to_do == 1:
        test = read_file('in.txt')
        c = Affine.encrypt(test, get_key_affine('key.txt', ALPHABET))
        print(c)
    elif which_cipher == 2 and what_to_do == 2:
        test = read_file('in.txt')
        c = Affine.decrypt(test, get_key_affine('key.txt', ALPHABET))
        print(c)
    elif which_cipher == 3 and what_to_do == 1:
        test = read_file('in.txt')
        c = Substitution.encrypt(test, get_key_substitution('key.txt'))
        print(c)
    elif which_cipher == 3 and what_to_do == 2:
        test = read_file('in.txt')
        c = Substitution.decrypt(test, get_key_substitution('key.txt'))
        print(c)
    elif which_cipher == 4 and what_to_do == 1:
        print('Нихуя нет')
    elif which_cipher == 4 and what_to_do == 2:
        print('Нихуя нет')
    elif which_cipher == 5 and what_to_do == 1:
        test = read_file('in.txt')
        c = ColumnarTransposition.encrypt(test, get_key_columnar('key.txt'))
        print(c)
    elif which_cipher == 5 and what_to_do == 2:
        test = read_file('in.txt')
        c = ColumnarTransposition.decrypt(test, get_key_columnar('key.txt'))
        print(c)
    elif which_cipher == 6 and what_to_do == 1:
        test = read_file('in.txt')
        c = Vigenere.encrypt(test, get_key_vigenere('key.txt'))
        print(c)
    elif which_cipher == 6 and what_to_do == 2:
        test = read_file('in.txt')
        c = Vigenere.decrypt(test, get_key_vigenere('key.txt'))
        print(c)
