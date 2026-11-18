🔧 Tool API
===========

Function-level documentation for the registered tools, grouped by the same clinical or
functional categories as :doc:`../tools/index`. Each page documents the modules and
functions behind a category, including their signatures, accepted inputs, and outputs.
Survey-only categories (fluoroscopy, nuclear medicine) have no callable API; their tool
pages record the candidate survey.

.. list-table::
   :header-rows: 1

   - - Page
     - CLI category
     - Package
   - - :doc:`medai`
     - ``medai``
     - ``caresmcp.medai``
   - - :doc:`tools`
     - ``image``
     - ``caresmcp.tools``
   - - :doc:`ct`
     - ``ct``
     - ``caresmcp.ct``
   - - :doc:`dsa`
     - ``dsa``
     - ``caresmcp.dsa``
   - - :doc:`mammography`
     - ``mammography``
     - ``caresmcp.mammography``
   - - :doc:`nuclearmed`
     - —
     - ``caresmcp.nuclearmed``
   - - :doc:`pathology`
     - ``pathology``
     - ``caresmcp.pathology``
   - - :doc:`ultrasound`
     - ``ultrasound``
     - ``caresmcp.ultrasound``
   - - :doc:`xray`
     - ``xray``
     - ``caresmcp.xray``

.. toctree::
   :hidden:

   medai
   tools
   ct
   dsa
   mammography
   nuclearmed
   pathology
   ultrasound
   xray
