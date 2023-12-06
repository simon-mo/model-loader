list:
    @just --list

build:
    maturin develop

test: build
    python app.py