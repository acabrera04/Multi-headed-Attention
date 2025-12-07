CC = gcc
CFLAGS = -g
LIBS = -lm

SRC_DIR = src
WORK_DIR = work

SERIAL_SRCS = $(SRC_DIR)/serial_attention.c $(SRC_DIR)/serial_load_model.c $(SRC_DIR)/load_tokens.c
SERIAL_TARGET = $(WORK_DIR)/serial_attention
MODEL_BIN = $(WORK_DIR)/gpt2_124m.bin
TOKENS_BIN = $(WORK_DIR)/tokens.bin

serial: $(SERIAL_TARGET)

$(WORK_DIR):
	mkdir -p $(WORK_DIR)

$(MODEL_BIN): | $(WORK_DIR)
	cd $(SRC_DIR) && python3 serialize_model.py

$(TOKENS_BIN): | $(WORK_DIR)
	cd $(SRC_DIR) && python3 tokenizer.py "hello world"

$(SERIAL_TARGET): $(SERIAL_SRCS) $(MODEL_BIN) $(TOKENS_BIN)
	@mkdir -p $(WORK_DIR)
	$(CC) $(CFLAGS) $(SERIAL_SRCS) -o $(SERIAL_TARGET) $(LIBS)

clean:
	rm -f $(SERIAL_TARGET) $(MODEL_BIN) $(TOKENS_BIN)
