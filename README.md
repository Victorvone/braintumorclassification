<!-- ABOUT THE PROJECT -->
## About The Project

[![Frontend]](braintumorclassification/screenshots/title_picture.png)

Convolutional networks have been widely used for the classification of objects in pictures. The use of such algorithms for clinical
decision assistants have been on the rise with models performing on par or better than humans in terms of accuracy for diagnosing diseases.
However, a frequent concern raised by clinicians relates to the missing explainability of this technology.

This private project aims to provide an MVP solution to this problem, adding multiple visualization techniques to the classification made
by our model.

Multiple pre-trained models were tested out via transfer-learning - namely VGG16, EfficientNetV2B3, InceptionNet and GoogleNet.
The model used in the end was EfficientNetV2B3, which reached an accuracy of 97% on our test set. This model was chosen as it represented
the best trade-off between accuracy and model size for us. Needless to mention the relevance of accuracy, model-size was an important
metric for us, since we used MLFlow as version control system for our model, which had a limit of upload-size.

The visualization techniques used in the project were grad-cam, activation visualizations, vanillagrad and occlusion sensitivity maps.
Grad-cam turned out to provide us the best results in terms of interpretability.

- Main packages used
- Logic to make the API + Logic to train the model, to load and save it.
- API & Frontend
-



<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- CONTACT -->
## Contributors

Victor von Eisenhart-Rothe - [LinkedIn](https://www.linkedin.com/in/victor-von-eisenhart-rothe/) - https://github.com/Victorvone
Aurélien Biais - [LinkedIn](https://www.linkedin.com/in/aur%C3%A9lien-biais-a41360a3/) - https://github.com/abiais
Ivan Andjelkovic - [LinkedIn](https://www.linkedin.com/in/ivan-andjelkovic-b6427029/) - https://github.com/IvanAndjelkovic
Aydoğan Avcıoğlu - [LinkedIn](https://www.linkedin.com/in/aydo%C4%9Fan-avc%C4%B1o%C4%9Flu-891466173/) - https://github.com/aydogan22

<p align="right">(<a href="#readme-top">back to top</a>)</p>
