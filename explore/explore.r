#===============================================================================#
# EXPLORE CFPB COMPLAINTS DATA
#
# Cecile Murray
#===============================================================================#

libs <- c("here",
          "tidyverse",
          "magrittr",
          "purrr",
          "knitr", 
          "kableExtra",
          "janitor")
lapply(libs, library, character.only = TRUE)

# fill colors
blue_fill <- "#217FBE"

#===============================================================================#
# LOAD DATA
#===============================================================================#

raw <- read_csv("../data/complaints.csv")

# filter to only complaints with text 
complaints <- raw %>% 
  clean_names() %>% 
  filter(!is.na(consumer_complaint_narrative),
         consumer_complaint_narrative != "")

narrative_stats <- read_csv("../data/complaint_stats.csv")

#===============================================================================#
# CROSSTABS
#===============================================================================#

# some crosstabs - looking good on missing values, generally
complaints %>% 
  tabyl(consumer_consent_provided) %>%
  mutate(percent = scales::number(percent, accuracy = 0.01))
complaints %>%
  tabyl(company_response_to_consumer) %>%
  mutate(percent = scales::number(percent, accuracy = 0.01))
complaints %>% 
  tabyl(company_public_response) %>%
  mutate(percent = scales::number(percent, accuracy = 0.01)) %>% View()

complaints %>% tabyl(company_public_response,
                     company_response_to_consumer) %>%
  mutate(percent = scales::number(percent, accuracy = 0.01)) %>% View()

# how many states does this affect?
complaints %>% 
  tabyl(state) %>% 
  mutate(percent = scales::number(percent, accuracy = 0.01))
# FL disproportionately represented

# number of unique companies: 4500
length(unique(complaints$company))

# mostly people are complaining about credit bureaus (30% of complaints)
complaints %>% 
  tabyl(company) %>% 
  mutate(percent = scales::number(percent, accuracy = 0.01)) %>% 
  arrange(-n) %>% 
  View()

# specifically credit reporting is the issue
complaints %>%
  tabyl(sub_product) %>% 
  mutate(percent = scales::number(percent, accuracy = 0.01)) %>% 
  arrange(-n) %>% 
  View()

min(complaints$date_received)
max(complaints$date_received)

# how many complaints
complaints %>% 
  filter(date_received < as.Date("2020-01-01")) %>% 
  nrow()

#===============================================================================#
# CHARTS FOR PROPOSAL
#===============================================================================#


# plot the class distribution for company response
complaints %>% 
  tabyl(company_response_to_consumer) %>% 
  ggplot(aes(x = reorder(company_response_to_consumer, percent),
             y = percent)) +
  geom_col(fill = blue_fill) +
  scale_x_discrete(labels = function(x) str_wrap(x, width = 10)) +
  scale_y_continuous(labels = scales::percent) +
  labs(title = "Most complaints received a response with explanation",
       subtitle = "Share of complaints by company response type",
       x = "Company response type",
       y = "% complaints",
       caption = "Source: CFPB Consumer Complaints Database")

# ggsave("explore/plots/complaints_by_company_response.png")

# plot class distribution for products
complaints %>% 
  tabyl(product) %>% 
  ggplot(aes(x = reorder(product, percent),
             y = percent)) +
  geom_col(fill = blue_fill) +
  scale_x_discrete(labels = function(x) str_wrap(x, width = 45)) +
  scale_y_continuous(labels = scales::percent) +
  coord_flip() +
  labs(title = "Consumer credit and debt collection products receive 
the most complaints",
       subtitle = "Share of complaints by product type",
       x = "Product",
       y = "% complaints",
       caption = "Source: CFPB Consumer Complaints Database")

# ggsave("plots/complaints_by_product.png",
#        width = 8, height = 5, units = "in")

# plot class distribution for issues
complaints %>% 
  tabyl(issue) %>% 
  filter(percent > 0.01) %>% 
  ggplot(aes(x = reorder(issue, percent),
             y = percent)) +
  geom_col(fill = blue_fill) +
  scale_x_discrete(labels = function(x) str_wrap(x, width = 45)) +
  scale_y_continuous(labels = scales::percent) +
  coord_flip() +
  labs(title = "Consumer credit products receive the most complaints",
       subtitle = "% of complaints by issue for issues receiving > 1% of complaints",
       x = "Issue",
       y = "% complaints",
       caption = "Source: CFPB Consumer Complaints Database")

# ggsave("plots/complaint_share_by_issue.png",
#        width = 8, height = 5, units = "in")

# plot companies complained about
complaints %>% 
  tabyl(company) %>% 
  filter(percent >= 0.02) %>% 
  ggplot(aes(x = reorder(company, percent),
             y = percent)) +
  geom_col(fill = blue_fill) +
  scale_x_discrete(labels = function(x) str_wrap(x, width = 20)) +
  scale_y_continuous(labels = scales::percent) +
  coord_flip() +
  labs(title = "The big 3 credit bureaus received one-third of complaints",
       subtitle = "Share of complaints for companies receiving >2% of complaints",
       x = "Company name",
       y = "% complaints",
       caption = "Source: CFPB Consumer Complaints Database")

# ggsave("explore/plots/complaints_by_company.png")

# plot complaints over time
complaints %>% 
  ggplot(aes(x = date_received)) +
  geom_vline(xintercept = as.Date("2016-01-01"),
             color = "black") +
  geom_vline(xintercept = as.Date("2020-01-01"),
             color = "black") +
  geom_freqpoly(color = blue_fill) +
  scale_x_date() +
  scale_y_continuous(labels = scales::number) +
  labs(title = "Complaints fluctuate over time",
       subtitle = "Number of complaints received over time",
       x = "Date complaint received by CFPB",
       y = "# complaints",
       caption = "Source: CFPB Consumer Complaints Database") 

ggsave("plots/complaints_over_time_with_bars.png")


# summary of narrative length (# characters)
char_summary <- narrative_stats %>% 
  summarize(
    avg = mean(complaint_char_len),
    median = median(complaint_char_len)
  )

# visualize distribution of review length (# characters)
narrative_stats %>% 
  ggplot(aes(x = complaint_char_len)) +
  geom_freqpoly(stat = "density", color = blue_fill) +
  scale_y_continuous(labels = scales::percent) 

# do longer complaints generally get specific response types?
# seems like not really
narrative_stats %>% 
  left_join(
    select(complaints, complaint_id, company_response_to_consumer),
    by = "complaint_id"
  ) %>% 
  ggplot(aes(x = company_response_to_consumer,
             y = complaint_char_len)) +
  geom_boxplot()

# what about by product? some variation here but not huge
narrative_stats %>% 
  left_join(
    select(complaints, complaint_id, product),
    by = "complaint_id"
  ) %>% 
  group_by(product) %>% 
  summarize(
    char_len_median = median(complaint_char_len),
    char_len_avg = mean(complaint_char_len),
    char_len_sd = sd(complaint_char_len)
  ) %>% 
  mutate(
    char_len_sd_low = char_len_avg - char_len_sd,
    char_len_sd_high = char_len_avg + char_len_sd
  ) %>% 
  ggplot() +
  geom_point(aes(x = product,
                    y = char_len_avg)) +
  geom_errorbar(aes(x = product,
                    ymin = char_len_sd_low,
                    ymax = char_len_sd_high)) +
  scale_x_discrete(labels = function(x) str_wrap(x, width = 30)) +
  coord_flip() 

#===============================================================================#
# GET A COMPLAINT
#===============================================================================#

complaints %>% 
  select(complaint_id, company, date_received, consumer_complaint_narrative) %>% 
  filter(str_detect(company, "EQUIFAX")) %>% 
  sample_frac(0.1) %>% 
  View()

c <- complaints %>% 
  filter(complaint_id == "1291009")
c$consumer_complaint_narrative
