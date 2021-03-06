import setuptools

if __name__ == "__main__":
    setuptools.setup(
        name='gauss_fit',
        version="0.1.1",
        description='',
        author='Sebastian Dick',
        author_email='sebastian.dick@stonybrook.edu',
        url="",
        license='BSD-3C',
        packages=setuptools.find_packages(),
        python_requires='>=3.6',
        install_requires=[
            'numpy',
            'sympy',
            'scipy',
            'pandas',
            'matplotlib',
        ],
        extras_require={
            'docs': [
                'sphinx==1.2.3',  # autodoc was broken in 1.3.1
                'sphinxcontrib-napoleon',
                'sphinx_rtd_theme',
                'numpydoc',
            ],
            'tests': [
                'pytest',
                'pytest-cov',
                'pytest-pep8',
                'tox',
            ],
        },

        tests_require=[
            'pytest',
            'pytest-cov',
            'pytest-pep8',
            'tox',
        ],

        classifiers=[
            'Development Status :: 4 - Beta',
            'Intended Audience :: Science/Research',
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3',
        ],
        zip_safe=True,
    )
