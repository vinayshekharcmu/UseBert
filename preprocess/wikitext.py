from config import *
import argparse
import os

def write_to_file(file_path, lines):
    with open(file_path, "w") as f:
        for line in lines:
            f.write(line)
            f.write("\n")

def read_wiki_file(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
    lines = filter(lambda x: len(x) != 1, lines)
    clean_lines = []
    for line in lines:
        split_lines = line.split("\n")
        for split_line in split_lines:
            clean_lines.extend(split_line.split("."))
    return clean_lines

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wiki-input-path")
    parser.add_argument("--wiki-output-path")
    args = parser.parse_args()
    wiki_train_lines = read_wiki_file(os.path.join(args.wiki_input_path,  WIKI_TRAIN))
    wiki_test_lines = read_wiki_file(os.path.join(args.wiki_input_path, WIKI_TEST))

    write_to_file(os.path.join(args.wiki_output_path,  WIKI_TRAIN), wiki_train_lines)
    write_to_file(os.path.join(args.wiki_output_path, WIKI_TEST), wiki_test_lines)




