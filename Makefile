CC = gcc
CFLAGS = -g
LIBS = -lm

SRC_DIR = src
WORK_DIR = work

SERIAL_SRCS = $(SRC_DIR)/serial_attention.c $(SRC_DIR)/serial_load_model.c $(SRC_DIR)/load_tokens.c
SERIAL_TARGET = $(WORK_DIR)/serial_attention

serial: $(SERIAL_TARGET)

$(SERIAL_TARGET): $(SERIAL_SRCS)
	@mkdir -p $(WORK_DIR)
	$(CC) $(CFLAGS) $^ -o $(SERIAL_TARGET) $(LIBS)

clean:
	rm -f $(SERIAL_TARGET)
