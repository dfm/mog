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

    python setup.py build_ext --inplace

Then, you can run the demo (as long as you have matplotlib installed):

::

    python demo.py

It will save a plot called ``demo.png`` that should look something like:

.. image:: https://raw.github.com/dfm/mog/master/demo-result.png