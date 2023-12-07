list:
    @just --list

build:
    maturin develop

build-release:
    maturin develop --release

test: build
    python app.py

bench: build-release
    python app.py