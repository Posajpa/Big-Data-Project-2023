### Social media sentiment analysis

---

***Abstract***

This project focuses on developing a data processing system for analyzing social media data, specifically tweets. The goal is to extract valuable insights and perform sentiment analysis to understand public opinion and trends. Through data exploration and the implementation of a robust pipeline, the project has successfully collected and processed a large volume of tweets, conducted sentiment analysis, and stored the processed data. The system enables users to gain valuable insights from social media data for various applications such as market research and brand monitoring.

---

#### Technologies

+ Spark
+ MongoDB
+ Docker with compose
+ Flask

---

#### Architecture

![Alt Text](readme_files/diagram.png)


---

#### How to run

The project comes with a `docker-compoe.yml` that you can use to create the image and run the container.
```bash0
docker-compose build # creation of the image (it might take some minutes)
docker-compose up   # running the container
```

Once it is done, you can navigate to http://127.0.0.1:5000/ using your browser.
and from there the interface will guide you.
