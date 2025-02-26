# Hi YC!

This is a cleaned and simplified version of the pipeline that would be used in MimicMol. There is also a flowchart of what is going on and a brief slidedeck in the repo as well. 

To use this code, you would need to input an OpenAI API key (or similar). I left mine out for privacy reasons.

Below is the result of one of the unit tests we conducted (for CBL-2) when evaluating this pipeline, of which there is more detail in the written application. I've also added both starting datasets, which were downloaded from ChemBL. 

For clarity, Assay 1 refers to "CBL-2.csv", and Assay 2 refers to "CBL-2-B.csv"

| Header   | Value  |
|----------|--------|
| Number of Generated Molecules | 10000 |
| Number of Generated Molecules, filtered for structural feasibility | 9000 |





| Header | Value | std-value | p-value |
|----------|----------|----------|----------|
| Average -log(Bioactivity) for Assay 1  | -6.5004   |  1.613  | |
| Average -log(Bioactivity) for Generated Molecules (as predicted by Model 1)  | -6.7112   | | 1.46 * 10^-35 |
| Average -log(Bioactivity) for Assay 2   | -5.4197    | 1.231  | |
| Average -log(Bioactivity) for Generated Molecules (as predicted by Model 2)   | -6.1315 |    | 1.4 x 10^-656 |
| Average -log(Bioactivity) for Assay 1 (as predicted by Model 2)   | -5.941  |  1.613  | |
| Average -log(Bioactivity) for Generated Molecules (as predicted by Model 2)  |  -6.1315  |    | 6 * 10^-88700 |


Let me know if you have any questions! 

Austin