Mixtures of Gaussians
=====================

By Dan Foreman-Mackey


Install
-------

Just run:

::

    python setup.py install

Then try:

::

    cd;python -c "import mog"


Demo
----

To try the demo, you need to build the C-extension "in place":

::

    python setup.pt build_ext --inplace

Then, you can run the demo and it will save a plot called ``demo.png`` if
you have ``matplotlib`` installed:

::

    python demo.py
