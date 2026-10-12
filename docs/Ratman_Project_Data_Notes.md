# Ratman project data notes

The application consumes georeferenced LAS/LAZ exports rather than raw sensor files. CloudStation remains responsible for sensor and trajectory processing, and the project must confirm appropriate underwater refraction correction before geometric object measurements are trusted.

Useful input groups include XYZ geometry, intensity, return number and number of returns, scan angle, optional range, GPS time, source/strip identifiers and available full-waveform information. The [sensor export documentation](https://helpcenter.yellowscan.com/knowledge/what-are-the-point-cloud-file-formats-that-i-can-export) describes LAS/LAZ and configurable ASCII exports. Actual availability must be checked in the supplied files.

Recorded intensity is sensor and processing dependent; a 16-bit storage scale does not make it calibrated reflectance. Depth, water clarity, propagation path, incidence angle and acquisition settings can affect it. Return numbers summarise discrete detected returns and must not be hard-coded as physical water-surface, target or seabed identities.

LAS classification is a prior processing result, not automatically the project target label or an instance annotation. Unknown user-data semantics should be confirmed before use. Source/strip identifiers are valuable for provenance and independent split design. RGB should not be assumed to represent reliable underwater object appearance.

The prototype calculates local geometric and relative-intensity features but does not reconstruct a validated seabed model, compensate radiometric attenuation or decode full waveform. It records waveform indicators without claiming that the samples are available and decodable.

The supplied report's scenario is approximately 2 m below the sea surface. This describes water depth, not an assumed 2 m object size. Target categories, size ranges, point densities and background composition still need specification.

The Chinese data report is included in `preparation/` with project names updated; these English notes summarise its data implications for the current prototype.
