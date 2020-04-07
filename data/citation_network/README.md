# Aminer Citation Network
The citation netwrok data is extracted from [Aminer](https://aminer.org/aminernetwork), which includes content of paper information, paper citation, author information and collaboration. We selected papers from four research domains, i.e., Data Mining (**DM**), Computer Vision (**CV**), Natural Language Processing (**NLP**) and Database (**DB**). Specifically, three top venues for each research area are selected. Each author is labeled with the area with the majority of his/her publications.

### Description
| file               | n_node   | n_edge                   | node_type                                     |
| ------------------ | -------- | -------------| --------------- |
| author-author.pt   | 27155 | 108330  | (author, author) |
| paper-author.pt    | 52303 | 17040+27155   | (paper, author)  |
| paper-paper.pt     | 109502| 46707   | (paper, paper)   |
