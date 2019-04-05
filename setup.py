from setuptools import setup, find_packages


PACKAGE_NAME = 'timelapsetracking'


"""
Notes:
MODULE_VERSION is read from timelapsetracking/version.py.
See (3) in following link to read about versions from a single source
https://packaging.python.org/guides/single-sourcing-package-version/#single-sourcing-the-version
"""

MODULE_VERSION = ""
exec(open(PACKAGE_NAME + "/version.py").read())


def readme():
    with open('README.md') as f:
        return f.read()


test_deps = ['pytest', 'pytest-cov']

lint_deps = ['flake8']
interactive_dev_deps = [
    # -- Add libraries/modules you want to use for interactive
    # -- testing below (e.g. jupyter notebook).
    # -- E.g.
    # 'matplotlib>=2.2.3',
    # 'jupyter',
    # 'itkwidgets==0.12.2',
    # 'ipython==7.0.1',
    # 'ipywidgets==7.4.1'
]
all_deps = [*test_deps, *lint_deps, *interactive_dev_deps]

extras = {
    'test_group': test_deps,
    'lint_group': lint_deps,
    'interactive_dev_group': interactive_dev_deps,
    'all': all_deps
}

setup(name=PACKAGE_NAME,
      version=MODULE_VERSION,
      description='Tracking code and scripts for the timelapse analysis project',
      long_description=readme(),
      author='Jianxu Chen',
      author_email='jianxuc@alleninstitute.org',
      license='Allen Institute Software License',
      packages=find_packages(exclude=['tests', '*.tests', '*.tests.*']),
      python_requires='>=3.6',
      entry_points={
          "console_scripts": [
              "my_example={}.bin.my_example:main".format(PACKAGE_NAME)
          ]
      },
      install_requires=[
          'numpy',
          'pandas',
          'scipy',
      ],

      # For test setup. This will allow JUnit XML output for Jenkins
      setup_requires=['pytest-runner'],
      tests_require=test_deps,

      extras_require=extras,
      zip_safe=False
      )
