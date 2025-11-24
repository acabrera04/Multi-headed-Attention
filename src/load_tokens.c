#include <stdio.h>
#include <stdlib.h>

int main() {
    FILE *f = fopen("../work/tokens.bin", "rb");

    // get file size (token is int)
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    int num_tokens = file_size / sizeof(int);
    printf("tokens.bin is %d tokens long\n", num_tokens);

    int *tokens = (int *)malloc(num_tokens * sizeof(int));

    size_t read_count = fread(tokens, sizeof(int), num_tokens, f);
    if (read_count != num_tokens) {
        fprintf(stderr, "Error reading tokens: expected %d, got %zu\n", num_tokens, read_count);
        free(tokens);
        fclose(f);
        return EXIT_FAILURE;
    }

    printf("tokens: [");
    for (int i = 0; i < num_tokens; i++) {
        printf("%d", tokens[i]);
        if (i < num_tokens - 1) {
            printf(", ");
        }
    }
    printf("]\n");

    free(tokens);
    fclose(f);
    return EXIT_SUCCESS;

    // TODO return the tokens in a function
}
