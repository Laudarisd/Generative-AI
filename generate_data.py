import pandas as pd

data = [
    {"Question": "Hello, how can I assist you today?", "Answer": "Hi! I'm here to help you with anything you need. What can I do for you?"},
    {"Question": "What perfumes do you recommend for women?", "Answer": "I recommend trying Chanel No. 5, Dior J'adore, or Marc Jacobs Daisy for women."},
    {"Question": "Can you suggest a perfume for a formal event?", "Answer": "For a formal event, you might like Yves Saint Laurent Black Opium or Chanel Coco Mademoiselle."},
    {"Question": "What are the best-selling perfumes this season?", "Answer": "The best-selling perfumes this season include Gucci Bloom, Tom Ford Black Orchid, and Jo Malone London."},
    {"Question": "How can I track my order?", "Answer": "You can track your order by logging into your account and checking the 'Order Status' section."},
    {"Question": "What is the return policy for perfumes?", "Answer": "Our return policy allows returns within 30 days of purchase, provided the perfume is unopened."},
    {"Question": "Can I get a discount on bulk purchases?", "Answer": "Yes, we offer discounts on bulk purchases. Please contact our sales team for more details."},
    {"Question": "How do I know if a perfume is original?", "Answer": "To ensure a perfume is original, buy from reputable retailers and check for authenticity seals."},
    {"Question": "Can you suggest a perfume for men?", "Answer": "For men, I suggest Dior Sauvage, Bleu de Chanel, or Giorgio Armani Acqua di Gio."},
    {"Question": "What is the difference between Eau de Parfum and Eau de Toilette?", "Answer": "Eau de Parfum has a higher concentration of fragrance oils and lasts longer than Eau de Toilette."},
    {"Question": "How long does the fragrance last?", "Answer": "The fragrance typically lasts between 4 to 8 hours, depending on the type and ingredients."},
    {"Question": "Can I get a sample before purchasing?", "Answer": "Yes, we offer samples for many of our perfumes. Please visit our samples section on the website."},
    {"Question": "How should I store my perfume?", "Answer": "Store your perfume in a cool, dry place away from direct sunlight to maintain its quality."},
    {"Question": "What ingredients are in this perfume?", "Answer": "You can find the ingredients list on the product page or packaging of the perfume."},
    {"Question": "Are there any perfumes without alcohol?", "Answer": "Yes, we have a selection of alcohol-free perfumes. Please check our 'Alcohol-Free' section."},
    {"Question": "How can I change my shipping address?", "Answer": "You can change your shipping address by going to your account settings and updating your information."},
    {"Question": "What payment methods do you accept?", "Answer": "We accept all major credit cards, PayPal, and other secure payment methods."},
    {"Question": "Can I cancel my order after placing it?", "Answer": "Orders can be canceled within an hour of placing them. Please contact our support team for assistance."},
    {"Question": "Do you offer gift wrapping services?", "Answer": "Yes, we offer gift wrapping services at checkout. You can select this option when placing your order."},
    {"Question": "Can you recommend a perfume for a summer day?", "Answer": "For a summer day, try light and fresh scents like Dolce & Gabbana Light Blue or Versace Bright Crystal."},
    {"Question": "How can I join your loyalty program?", "Answer": "You can join our loyalty program by signing up on our website. Earn points with every purchase!"},
    {"Question": "Do you ship internationally?", "Answer": "Yes, we ship to many countries worldwide. Please check our shipping policy for more details."},
    {"Question": "What should I do if my perfume arrives damaged?", "Answer": "If your perfume arrives damaged, please contact our support team immediately for a replacement or refund."},
    {"Question": "Can I return a used perfume?", "Answer": "Unfortunately, we cannot accept returns on used perfumes due to hygiene reasons."},
    {"Question": "Do you offer express shipping?", "Answer": "Yes, we offer express shipping options at checkout for faster delivery."},
    {"Question": "How do I apply perfume for the best results?", "Answer": "Apply perfume to pulse points like wrists, neck, and behind ears for long-lasting fragrance."},
    {"Question": "What are niche perfumes?", "Answer": "Niche perfumes are exclusive, high-quality fragrances made by specialized perfumeries."},
    {"Question": "Do you have perfumes suitable for sensitive skin?", "Answer": "Yes, we offer perfumes formulated for sensitive skin. Please check our 'Sensitive Skin' section."},
    {"Question": "Can I combine different perfumes?", "Answer": "Yes, you can layer perfumes to create a unique scent. Start with lighter scents and add stronger ones."},
    {"Question": "What size bottles do you offer?", "Answer": "We offer various sizes, including 30ml, 50ml, and 100ml bottles."},
    {"Question": "How often do you release new perfumes?", "Answer": "We release new perfumes seasonally. Stay updated by subscribing to our newsletter."},
    {"Question": "Are your perfumes cruelty-free?", "Answer": "Yes, all our perfumes are cruelty-free and not tested on animals."},
    {"Question": "What is the best way to sample multiple perfumes?", "Answer": "You can order a sample set from our website to try multiple perfumes before buying full-sized bottles."},
    {"Question": "Can I personalize my perfume bottle?", "Answer": "Yes, we offer personalization services for select perfume bottles. Please check our personalization section."},
    {"Question": "Do you have a physical store?", "Answer": "Yes, we have physical stores. Please check our store locator for the nearest location."},
    {"Question": "Can I get recommendations based on my previous purchases?", "Answer": "Yes, log into your account to get personalized recommendations based on your purchase history."},
    {"Question": "Hi, what perfumes do you recommend for teenagers?", "Answer": "For teenagers, I recommend light and fresh scents like Marc Jacobs Daisy or Calvin Klein CK One."},
    {"Question": "Hello, can you suggest a gift for a perfume lover?", "Answer": "For a perfume lover, I suggest a gift set featuring a variety of mini perfumes or a personalized bottle."},
    {"Question": "Hi, do you have any eco-friendly perfumes?", "Answer": "Yes, we offer a range of eco-friendly perfumes. Please check our 'Eco-Friendly' section."},
    {"Question": "Hello, what are the ingredients in natural perfumes?", "Answer": "Natural perfumes are made with essential oils, plant extracts, and other natural ingredients. You can find the full list on the product page."},
    {"Question": "Hi, can you help me find a perfume similar to my favorite scent?", "Answer": "Yes, please tell me the name of your favorite scent, and I'll suggest some similar options."},
    {"Question": "Hello, what are your most popular unisex perfumes?", "Answer": "Our most popular unisex perfumes include Jo Malone London Wood Sage & Sea Salt, and Le Labo Santal 33."},
    {"Question": "Hi, how can I leave a review for a perfume?", "Answer": "You can leave a review by logging into your account, going to the product page, and submitting your review."},
    {"Question": "Hello, what is your best perfume for winter?", "Answer": "For winter, I recommend warm and spicy scents like Tom Ford Tobacco Vanille or Viktor & Rolf Spicebomb."},
    {"Question": "Hi, can you recommend a long-lasting perfume?", "Answer": "For a long-lasting perfume, try Lancome La Vie Est Belle or Paco Rabanne 1 Million."},
    {"Question": "Hello, how do I contact customer service?", "Answer": "You can contact our customer service through the 'Contact Us' page on our website or by calling our hotline."},
    {"Question": "Hi, do you have vegan perfumes?", "Answer": "Yes, we offer a selection of vegan perfumes. Please check our 'Vegan' section for more details."},
    {"Question": "Hello, can you suggest a perfume for a romantic evening?", "Answer": "For a romantic evening, try seductive scents like Chanel Chance or YSL Mon Paris."},
    {"Question": "Hi, how do I use a perfume sample?", "Answer": "Apply the perfume sample to your wrist or neck to test the fragrance. Samples are perfect for trying out a scent before buying a full bottle."},
    {"Question": "Hello, what perfumes are best for daytime wear?", "Answer": "For daytime wear, light and fresh scents like Versace Bright Crystal or Marc Jacobs Daisy are great options."},
    {"Question": "Hi, can you recommend a perfume for a birthday gift?", "Answer": "For a birthday gift, consider personalized perfumes or gift sets featuring popular scents like Chanel No. 5 or Dior J'adore."},
    {"Question": "Hello, do you offer free samples?", "Answer": "We offer free samples with certain purchases and promotions. Check our website for current offers."},
    {"Question": "Hi, what is your policy on exchanging perfumes?", "Answer": "Our exchange policy allows you to exchange unopened perfumes within 30 days of purchase. Contact our support team for assistance."},
    {"Question": "Hello, what are the top notes in this perfume?", "Answer": "The top notes of a perfume are the initial scents you smell when you first apply it. Check the product page for specific top notes information."},
    {"Question": "Hi, can you suggest a perfume for a casual outing?", "Answer": "For a casual outing, try light and airy scents like Dolce & Gabbana Light Blue or Burberry Brit."},
    {"Question": "Hello, how do I redeem a discount code?", "Answer": "You can redeem a discount code at checkout by entering the code in the designated field."},
    {"Question": "Hi, do you offer gift cards?", "Answer": "Yes, we offer gift cards in various amounts. You can purchase them on our website."},
    {"Question": "Hello, what are your store hours?", "Answer": "Our store hours vary by location. Please check our store locator on the website for specific hours."},
    {"Question": "Hi, can you help me choose a perfume based on my preferences?", "Answer": "Yes, please tell me about your fragrance preferences, and I'll suggest some options that match your taste."},
    {"Question": "Hello, do you have any limited edition perfumes?", "Answer": "Yes, we offer limited edition perfumes from time to time. Check our 'Limited Edition' section for current offerings."},
    {"Question": "Hi, what is the best way to layer perfumes?", "Answer": "Layering perfumes involves applying different scents in a sequence. Start with a base note, followed by a middle note, and finish with a top note for a unique fragrance."},
    {"Question": "Hello, how do I update my account information?", "Answer": "You can update your account information by logging into your account and editing your profile settings."},
    {"Question": "Hi, do you offer student discounts?", "Answer": "Yes, we offer student discounts. Please provide a valid student ID at checkout to receive the discount."},
    {"Question": "Hello, can you recommend a perfume for a beach day?", "Answer": "For a beach day, try fresh and aquatic scents like Davidoff Cool Water or Acqua di Gioia by Giorgio Armani."},
    {"Question": "Hi, what are the benefits of joining your loyalty program?", "Answer": "Joining our loyalty program offers benefits such as earning points on purchases, exclusive discounts, and early access to new releases."},
    {"Question": "Hello, do you have any perfumes with floral notes?", "Answer": "Yes, we have a wide range of perfumes with floral notes. Check our 'Floral' section for options."},
    {"Question": "Hi, can you suggest a perfume for a night out?", "Answer": "For a night out, try bold and alluring scents like Tom Ford Noir or Jean Paul Gaultier Scandal."},
    {"Question": "Hello, how do I subscribe to your newsletter?", "Answer": "You can subscribe to our newsletter by entering your email address in the subscription box on our website."},
    {"Question": "Hi, do you offer complimentary gift wrapping?", "Answer": "Yes, we offer complimentary gift wrapping on select items. You can choose this option at checkout."},
    {"Question": "Hello, what is your most popular perfume?", "Answer": "Our most popular perfume is Chanel No. 5, known for its timeless elegance and classic scent."},
    {"Question": "Hi, how do I apply a rollerball perfume?", "Answer": "Apply a rollerball perfume by rolling it onto pulse points like your wrists, neck, and behind your ears."},
    {"Question": "Hello, can you recommend a perfume for winter?", "Answer": "For winter, try warm and cozy scents like Viktor & Rolf Flowerbomb or Burberry London."},
    {"Question": "Hi, do you have any perfumes with woody notes?", "Answer": "Yes, we have perfumes with woody notes. Check our 'Woody' section for options like Tom Ford Oud Wood or Gucci Guilty."},
    {"Question": "Hello, how can I find out about upcoming sales?", "Answer": "Sign up for our newsletter or follow us on social media to stay updated on upcoming sales and promotions."},
    {"Question": "Hi, what are the middle notes in this perfume?", "Answer": "The middle notes, or heart notes, are the main body of the fragrance. Check the product page for specific middle notes information."},
    {"Question": "Hello, can you suggest a perfume for a special occasion?", "Answer": "For a special occasion, try luxurious and sophisticated scents like Creed Aventus or Hermès Terre d'Hermès."},
    {"Question": "Hi, do you offer free shipping?", "Answer": "We offer free shipping on orders over a certain amount. Check our shipping policy for more details."},
    {"Question": "Hello, how do I sign in to my account?", "Answer": "You can sign in to your account by clicking the 'Sign In' button on the top right corner of our website."},
    {"Question": "Hi, can you recommend a perfume for a spring day?", "Answer": "For a spring day, try fresh and floral scents like Marc Jacobs Daisy or Jo Malone Peony & Blush Suede."},
    {"Question": "Hello, what is the best way to store perfume samples?", "Answer": "Store perfume samples in a cool, dry place away from direct sunlight to maintain their quality."},
    {"Question": "Hi, do you offer corporate gifts?", "Answer": "Yes, we offer corporate gifts. Please contact our sales team for more details and options."},
    {"Question": "Hello, how do I reset my password?", "Answer": "You can reset your password by clicking the 'Forgot Password' link on the sign-in page and following the instructions."},
    {"Question": "Hi, what are the base notes in this perfume?", "Answer": "The base notes are the lasting scents that appear after the top and middle notes evaporate. Check the product page for specific base notes information."},
    {"Question": "Hello, can you recommend a perfume for a festive event?", "Answer": "For a festive event, try vibrant and joyful scents like Chanel Chance Eau Tendre or Viktor & Rolf Bonbon."},
    {"Question": "Hi, do you offer fragrance consultations?", "Answer": "Yes, we offer fragrance consultations. Please book an appointment through our website."},
    {"Question": "Hello, what are the benefits of using a solid perfume?", "Answer": "Solid perfumes are portable, easy to apply, and often have a longer shelf life. They are also a great option for travel."},
    {"Question": "Hi, do you have any perfumes with citrus notes?", "Answer": "Yes, we have perfumes with citrus notes. Check our 'Citrus' section for options like Dolce & Gabbana Light Blue or Clinique Happy."},
]

# Extend the dataset to reach at least 100 rows by duplicating and slightly modifying existing questions
additional_data = [
    {"Question": "Hello, what are the top perfumes for men?", "Answer": "For men, the top perfumes are Dior Sauvage, Bleu de Chanel, and Armani Code."},
    {"Question": "Hi, how can I earn points in the loyalty program?", "Answer": "You can earn points by making purchases, writing reviews, and referring friends."},
    {"Question": "Hello, do you have any perfumes inspired by nature?", "Answer": "Yes, we have nature-inspired perfumes like Jo Malone Wood Sage & Sea Salt and Maison Margiela Replica Beach Walk."},
    {"Question": "Hi, can you suggest a perfume for a graduation gift?", "Answer": "For a graduation gift, consider fresh and youthful scents like Marc Jacobs Daisy or Dolce & Gabbana Light Blue."},
    {"Question": "Hello, what are the latest perfume releases?", "Answer": "The latest releases include Chanel Gabrielle Essence, Yves Saint Laurent Libre, and Gucci Mémoire d'une Odeur."},
    {"Question": "Hi, can you recommend a perfume for an anniversary gift?", "Answer": "For an anniversary gift, try romantic and luxurious scents like Chanel Coco Mademoiselle or Dior J'adore."},
    {"Question": "Hello, do you offer perfume samples with every purchase?", "Answer": "Yes, we offer complimentary samples with every purchase, subject to availability."},
    {"Question": "Hi, what perfumes are popular among teenagers?", "Answer": "Popular perfumes among teenagers include Ariana Grande Cloud, Victoria's Secret Bombshell, and Hollister Wave."},
    {"Question": "Hello, can you suggest a perfume for a beach vacation?", "Answer": "For a beach vacation, try aquatic and fresh scents like Davidoff Cool Water or Escada Aqua Del Sol."},
    {"Question": "Hi, what are the best perfumes for evening wear?", "Answer": "For evening wear, try elegant and sophisticated scents like Tom Ford Black Orchid or Givenchy L'Interdit."},
    {"Question": "Hello, do you have any perfumes with gourmand notes?", "Answer": "Yes, we have perfumes with gourmand notes like Thierry Mugler Angel and Prada Candy."},
    {"Question": "Hi, how can I redeem my loyalty points?", "Answer": "You can redeem your loyalty points at checkout by selecting the option to apply points to your purchase."},
    {"Question": "Hello, what is the difference between parfum and cologne?", "Answer": "Parfum has a higher concentration of fragrance oils and lasts longer, while cologne is lighter and more refreshing."},
    {"Question": "Hi, can you recommend a perfume for a work environment?", "Answer": "For a work environment, try subtle and professional scents like Chanel Chance Eau Fraîche or Burberry Brit."},
    {"Question": "Hello, do you offer fragrance layering sets?", "Answer": "Yes, we offer fragrance layering sets that include complementary scents for a personalized fragrance experience."},
    {"Question": "Hi, what is the best way to apply perfume?", "Answer": "The best way to apply perfume is to spray it on pulse points like wrists, neck, and behind the ears."},
    {"Question": "Hello, can you recommend a perfume for a winter day?", "Answer": "For a winter day, try warm and cozy scents like Tom Ford Tobacco Vanille or Burberry London."},
    {"Question": "Hi, do you have any perfumes with musk notes?", "Answer": "Yes, we have perfumes with musk notes like Narciso Rodriguez For Her and Maison Francis Kurkdjian Baccarat Rouge 540."},
    {"Question": "Hello, how can I receive updates on new products?", "Answer": "You can receive updates on new products by subscribing to our newsletter and following us on social media."},
]

# Append the additional data to the main dataset to reach 100 rows
data.extend(additional_data)

# Convert to DataFrame and save as CSV
df = pd.DataFrame(data)
df.to_csv('perfume_customer_service.csv', index=False)