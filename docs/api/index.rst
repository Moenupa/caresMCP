🔧 Tool API
===========

Function-level documentation for the registered tools, grouped by the same clinical or
functional categories as :doc:`../coverage/index`. Each page documents the modules and
functions behind a category, including their signatures, accepted inputs, and outputs.
For choosing a tool, start with :doc:`../capabilities/index`; use this section when you
need the exact signature or return type.

Survey-only categories (fluoroscopy, nuclear medicine) have no callable API; their
coverage pages record the candidate survey.

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
   - - :doc:`mri`
     - ``mri``
     - ``caresmcp.mri``
   - - :doc:`nuclearmed`
     - ``nuclearmed``
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
   mri
   nuclearmed
   pathology
   ultrasound
   xray

.. rst-class:: generated-page-note

Auto-generated.
