import os
import requests


class EstateAPIs(object):
    def __init__(self):
        self.key = os.environ['ESTATE_API_KEY']
        self.base_urls = {
            'sale_query': "https://api.rentcast.io/v1/listings/sale",
            'rent_query': 'https://api.rentcast.io/v1/listings/rental/long-term'
        }
        
        self.headers = {
            "accept": "application/json",
            "X-Api-Key": self.key
        }
    
    def query_sale(self, query_configs):
        """
        Query the sale estate candidates from the estate API
        """
        import ipdb; ipdb.set_trace(context=20)
        # url = "https://api.rentcast.io/v1/listings/sale?city=Austin&state=TX&status=Active&limit=5"
        base_url = self.base_urls['sale_query']
        params = self.config_to_params(query_configs)

        response_json = requests.get(base_url, params=params, headers=self.headers).json()

        return response_json

    @staticmethod
    def config_to_params(query_configs):
        params = {}
        for key, value in query_configs.items():
            if query_configs[key] != 'None':
                params[key] = value

        return params

    def query_rent(self, query_configs):
        base_url = self.base_urls['rent_query']
        params = self.config_to_params(query_configs)

        response_json = requests.get(base_url, params=params, headers=self.headers).json()

        return response_json

    @staticmethod
    def filter_by_price(candidates, min_price, max_price):
        new_candidates = []
        for candidate in candidates:
            if candidate['price'] > min_price and candidate['price'] < max_price:
                new_candidates.append(candidate)

        return new_candidates
