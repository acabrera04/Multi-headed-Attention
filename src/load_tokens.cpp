#include <stdio.h>
#include <stdlib.h>
#include "load_tokens.h"

int *load_tokens(const char *filename, int *num_tokens)
{
    FILE *f = fopen(filename, "rb");
    if (!f)
    {
        printf("failed to open tokens file: %s\n", filename);
        return NULL;
    }

    // get file size (token is int)
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    *num_tokens = file_size / sizeof(int);

    int *tokens = (int *)malloc(*num_tokens * sizeof(int));

    size_t read_count = fread(tokens, sizeof(int), *num_tokens, f);
    if (read_count != (size_t)*num_tokens)
    {
        fprintf(stderr, "Error reading tokens: expected %d, got %zu\n", *num_tokens, read_count);
        free(tokens);
        fclose(f);
        return NULL;
    }

    fclose(f);
    return tokens;
}
