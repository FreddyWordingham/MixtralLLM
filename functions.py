import sys

import modal


stub = modal.Stub("example-hello-world")


@stub.function()
def foo(i):
    if i % 2 == 0:
        print("hello", i)
    else:
        print("world", i, file=sys.stderr)

    return i * i


@stub.local_entrypoint()
def main():
    # Call the function locally.
    print(foo.local(1000))

    # Call the function remotely.
    print(foo.remote(1000))

    # Parallel map.
    total = 0
    for ret in foo.map(range(20)):
        total += ret

    print(total)
