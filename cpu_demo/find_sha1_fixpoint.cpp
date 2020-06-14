#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <assert.h>
#include <math.h>

// note that this is in bytes (half the number of hex characters)
#define PREFIX_LEN 4
#define PREFIX_COUNTER_TYPE uint32_t

/* NOTE this program assumes the system is little-endian
function to test this from https://stackoverflow.com/a/1001373 that I used:

bool is_big_endian(void) {
    union {
        uint32_t i;
        char c[4];
    } bint = {0x01020304};

    return bint.c[0] == 1;
}

From what I can tell, CUDA operations are also little-endian, so I will not
bother with trying to adapt this program to work on big-endian systems.
(source: https://stackoverflow.com/questions/15356622/anyone-know-whether-nvidias-gpus-are-big-or-little-endian#15357410)
*/

uint32_t leftrotate(uint32_t a, uint32_t b) {
    uint32_t tmp = a << b;
    uint32_t lowbits = a >> (32 - b);
    return tmp | lowbits;
}

/* adapted from https://en.wikipedia.org/wiki/SHA-1#SHA-1_pseudocode */
void sha1(uint8_t* result, uint8_t* message, unsigned int n_bytes) {
    uint32_t
        h0 = 0x67452301,
        h1 = 0xEFCDAB89,
        h2 = 0x98BADCFE,
        h3 = 0x10325476,
        h4 = 0xC3D2E1F0;

    uint64_t m1 = 8 * n_bytes;

    int pad_bytes = 56 - (n_bytes % 64);
    if (pad_bytes <= 0) { // if = 0, need to increase so there's room for 0x80
        pad_bytes += 64;
    }
    int full_message_length = n_bytes + pad_bytes + 8;

    uint8_t* message_padded = (uint8_t*) calloc(full_message_length, sizeof(uint8_t));
    memcpy(message_padded, message, n_bytes);
    message_padded[n_bytes] = 0x80;

    for (int i = 0; i < 8; i++) {
        message_padded[n_bytes + pad_bytes + i] = ((uint8_t*)&m1)[ 7 - i ];
    }
    // NOTE: on a big-endian system this would be much easier:
    // *(uint64_t*)(message_padded + n_bytes + pad_bytes) = m1;

    uint32_t* w = (uint32_t*)malloc(80 * sizeof(uint32_t));
    for (int chunk = 0; chunk < full_message_length; chunk += 64) {
        for (int i = 0; i < 16; i++) {
            uint8_t* current_word = (uint8_t*)(w + i);
            for (int byte = 0; byte < 4; byte++) {
                current_word[3 - byte] = message_padded[ chunk + (4 * i) + byte ];
            }
        }
        // generating the words is also theoretically easier on a big-endian
        // system: memcpy(w, message_padded + chunk, 16 * sizeof(uint32_t));

        for (int i = 16; i < 80; i++) {
            w[i] = leftrotate(w[i-3] ^ w[i-8] ^ w[i-14] ^ w[i-16], 1);
        }

        uint32_t a = h0, b = h1, c = h2, d = h3, e = h4, f, k;
        for (int i = 0; i < 80; i++) {
            if (i < 20) {
                f = (b & c) | ((~b) & d);
                k = 0x5A827999;
            } else if (i < 40) {
                f = b ^ c ^ d;
                k = 0x6ED9EBA1;
            } else if (i < 60) {
                f = (b & c) | (b & d) | (c & d);
                k = 0x8F1BBCDC;
            } else {
                f = b ^ c ^ d;
                k = 0xCA62C1D6;
            }

            uint32_t temp = leftrotate(a, 5) + f + e + k + w[i];
            e = d;
            d = c;
            c = leftrotate(b, 30);
            b = a;
            a = temp;
        }

        h0 += a;
        h1 += b;
        h2 += c;
        h3 += d;
        h4 += e;
    }
    free(w);

    for (int i = 0; i < 4; i++) {
        result[ 3 - i] = h0 >> 8 * i;
        result[ 7 - i] = h1 >> 8 * i;
        result[11 - i] = h2 >> 8 * i;
        result[15 - i] = h3 >> 8 * i;
        result[19 - i] = h4 >> 8 * i;
    }
}

void print_hex(uint8_t* arr, int size) {
    for (int i = 0; i < size; i++) {
        printf("%02x", arr[i]);
    }
}

union PrefixCounter {
    PREFIX_COUNTER_TYPE n;
    uint8_t prefix[PREFIX_LEN];
};

int main(int argc, char** argv) {
    assert(sizeof(PREFIX_COUNTER_TYPE) == PREFIX_LEN * sizeof(uint8_t));

    PrefixCounter p;
    p.n = 0;

    uint8_t result[20];

    do {
        sha1(result, p.prefix, PREFIX_LEN);

        if (! memcmp(p.prefix, result, PREFIX_LEN * sizeof(uint8_t))) {
            print_hex(p.prefix, PREFIX_LEN);
            printf(" is a fixpoint!\n");
            break;
        } else if (p.n % 65536 == 0) {
            print_hex(p.prefix, PREFIX_LEN);
            printf(" is not a fixpoint\n");
        }

        p.n++;
    } while (p.n != 0);

    return 0;
}
