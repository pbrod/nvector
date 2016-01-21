from nvector._examples import getting_started_functional
__doc__ = getting_started_functional  # @ReservedAssignment


def write_readme(doc):

    with open('readme.txt', 'w') as fid:
        fid.write(doc)


if __name__ == '__main__':
    write_readme(__doc__)
