#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <assert.h>
#include <math.h>
#include <iostream>
#include <fstream>

// note that this is in bytes (half the number of hex characters)
#define PREFIX_LEN 3
#define PREFIX_COUNTER_TYPE uint32_t

#define DIGEST_LEN 20
#define HEXDIGEST_LEN 40

/* NOTE this program assumes the system is little-endian */

struct TreePuzzle {
    int num_layers;
    bool* digest_types;

    int* prefix_sizes;
    char** prefixes;

    int* suffix_sizes;
    char** suffixes;
};


// https://stackoverflow.com/a/5840160
std::ifstream::pos_type filesize(const char* filename)
{
    std::ifstream in(filename, std::ifstream::ate | std::ifstream::binary);
    return in.tellg();
}

void load_from_dir(struct TreePuzzle& target, const char* dir) {
    // if you're reading this function I'm sorry

    char* digest_bits_fname = (char*) malloc(strlen(dir) + 20);
    strcpy(digest_bits_fname, dir);
    strcat(digest_bits_fname, "digest_bits.txt");
    std::ifstream digest_bits_stream(digest_bits_fname);
    target.num_layers = 0;
    int capacity = 8;
    target.digest_types = (bool *) malloc(capacity * sizeof(bool));
    char bit;
    while (digest_bits_stream >> bit) {
        target.num_layers++;
        if (target.num_layers > capacity) {
            capacity <<= 1;
            target.digest_types = (bool*) realloc(target.digest_types,
                capacity * sizeof(bool));
        }
        if (bit == '1') {
            target.digest_types[target.num_layers - 1] = true;
        } else if (bit == '0') {
            target.digest_types[target.num_layers - 1] = false;
        } else {
            std::cerr << "unexpected bit in digest_bits.txt: " << bit << "\n";
            exit(1);
        }
    }
    free(digest_bits_fname);
    assert(target.num_layers > 0);

    target.prefix_sizes = (int *) malloc(target.num_layers * sizeof(int));
    target.prefixes = (char**) malloc(target.num_layers * sizeof(char*));

    target.suffix_sizes = (int *) malloc(target.num_layers * sizeof(int));
    target.suffixes = (char**) malloc(target.num_layers * sizeof(char*));

    for (int i = 0; i < target.num_layers; i++) {
        char prefix_fname[50];
        char suffix_fname[50];
        sprintf(prefix_fname, "%s/prefixes/%d.txt", dir, i);
        sprintf(suffix_fname, "%s/suffixes/%d.txt", dir, i);

        target.prefix_sizes[i] = filesize(prefix_fname);
        target.suffix_sizes[i] = filesize(suffix_fname);

        target.prefixes[i] = (char*) malloc(target.prefix_sizes[i]);
        target.suffixes[i] = (char*) malloc(target.suffix_sizes[i]);

        std::ifstream prefix_in, suffix_in;
        prefix_in.open(prefix_fname);
        suffix_in.open(suffix_fname);

        prefix_in.read(target.prefixes[i], target.prefix_sizes[i]);
        suffix_in.read(target.suffixes[i], target.suffix_sizes[i]);

        prefix_in.close();
        suffix_in.close();
    }
}
void free_tree_contents(struct TreePuzzle& t) {
    free(t.digest_types);
    for (int i = 0; i < t.num_layers; i++) {
        free(t.prefixes[i]);
        free(t.suffixes[i]);
    }
    free(t.prefixes);
    free(t.suffixes);
    free(t.prefix_sizes);
    free(t.suffix_sizes);
}
void print_tree_summary(struct TreePuzzle& t) {
    for (int i = 0; i < t.num_layers; i++) {
        printf("layer %d: prefix size %d at %p, suffix size %d at %p\n",
            i, t.prefix_sizes[i], t.prefixes[i],
            t.suffix_sizes[i], t.suffixes[i]);
    }
}

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

    uint32_t w[80];
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

    for (int i = 0; i < 4; i++) {
        result[ 3 - i] = h0 >> 8 * i;
        result[ 7 - i] = h1 >> 8 * i;
        result[11 - i] = h2 >> 8 * i;
        result[15 - i] = h3 >> 8 * i;
        result[19 - i] = h4 >> 8 * i;
    }
}


void hash_hex_digest_inplace(uint8_t* hash) {
    char output[HEXDIGEST_LEN + 1]; // extra space for null terminator
    for (int i = 0; i < DIGEST_LEN; i++) {
        snprintf(output + (2 * i), 3, "%02x", hash[i]);
        // writes 3 bytes: 2 bytes for the 2 hex digits and a null terminator
    }
    memcpy(hash, output, HEXDIGEST_LEN);
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
    struct TreePuzzle t;
    load_from_dir(t, argv[1]);

    assert(sizeof(PREFIX_COUNTER_TYPE) >= PREFIX_LEN * sizeof(uint8_t));

    PrefixCounter p;
    p.n = 0;

    int* layer_sizes = (int*) malloc(t.num_layers * sizeof(int));
    uint8_t** data_by_layer = (uint8_t**) malloc(t.num_layers * sizeof(uint8_t*));
    int* insert_sizes = (int*) malloc(t.num_layers * sizeof(int));

    for (int layer = 0; layer < t.num_layers; layer++) {
        if (layer == 0) {
            insert_sizes[layer] = 2 * PREFIX_LEN;
            // NOTE: assumes initial hash written to file is hex-digested
            // TODO: make this customizable in digest_bits.txt instead of the
            // final output (as what is desired there depends on the context)
        } else if (t.digest_types[layer - 1]) {
            insert_sizes[layer] = HEXDIGEST_LEN;
        } else {
            insert_sizes[layer] = DIGEST_LEN;
        }

        layer_sizes[layer] = t.prefix_sizes[layer] +
            t.suffix_sizes[layer] + insert_sizes[layer];
        data_by_layer[layer] = (uint8_t*) malloc(layer_sizes[layer]);

        memcpy(data_by_layer[layer],
            t.prefixes[layer], t.prefix_sizes[layer]);
        memcpy(data_by_layer[layer] + t.prefix_sizes[layer] + insert_sizes[layer],
            t.suffixes[layer], t.suffix_sizes[layer]);
    }

    uint8_t hash[HEXDIGEST_LEN]; // sometimes only first DIGEST_LEN bytes used
    do {
        memcpy(hash, p.prefix, PREFIX_LEN);

        hash_hex_digest_inplace(hash);
        // NOTE: assumes initial hash written to file is hex-digested
        // (see note above when computing insert_sizes[0])

        for (int layer = 0; layer < t.num_layers; layer++) {
            memcpy(data_by_layer[layer] + t.prefix_sizes[layer],
                hash, insert_sizes[layer]);
            sha1(hash, data_by_layer[layer], layer_sizes[layer]);

            if (t.digest_types[layer]) {
                hash_hex_digest_inplace(hash);
            }
        }

        if (! memcmp(p.prefix, hash, PREFIX_LEN)) {
            print_hex(p.prefix, PREFIX_LEN);
            printf(" is a fixpoint!\n");
            break;
        } else if (!(p.n & 0xffff)) {
            print_hex(p.prefix, PREFIX_LEN);
            printf(" is not a fixpoint\n");
        }

        p.n++;
    } while (p.n != 0);

    free_tree_contents(t);
    free(layer_sizes);
    free(data_by_layer);
    free(insert_sizes);

    return 0;
}
