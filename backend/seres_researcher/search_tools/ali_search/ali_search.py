import json
import os
from alibabacloud_iqs20241111 import models
from alibabacloud_iqs20241111.client import Client
from alibabacloud_tea_openapi import models as open_api_models
from typing import Optional, Dict, Any

class AliSearch:
    """
    Aliyun Search Retriever
    """
    def __init__(self, query, headers=None, query_domains=None):
        self.query = query
        self.headers = headers or {}
        self.query_domains = query_domains or None
        # Use hardcoded values from ali_search_tools.py for testing
        self.access_key_id = os.environ["AliSearch_KEY"]
        self.access_key_secret = os.environ["AliSearch_SECRET"]

        if not self.access_key_id or not self.access_key_secret:
            raise Exception("Aliyun Access Key ID or Secret not found. Please set ALI_ACCESS_KEY_ID and ALI_ACCESS_KEY_SECRET environment variables or pass them in headers.")

    def create_client(self) -> Client:
        config = open_api_models.Config(
            access_key_id=self.access_key_id,
            access_key_secret=self.access_key_secret
        )
        config.endpoint = 'iqs.cn-zhangjiakou.aliyuncs.com'
        return Client(config)

    def search(self, max_results=7, time_range: str = "OneMonth", industry: Optional[str] = None) -> list[dict[str, str]]:
        print(f"正在使用AliSearch搜索相关信息...")
        client = self.create_client()
        request = models.GenericSearchRequest(
            query=self.query,
            time_range=time_range,
            industry=industry
        )
        
        try:
            response = client.generic_search(request).body.to_map()
            # The ali_search_test.py output shows 'pageItems' as the key for results
            results = response.get("pageItems", [])
        except Exception as e:
            print(f"Error during AliSearch: {e}")
            return []

        search_results = []
        for result in results:
            # Assuming 'htmlTitle' is title, 'link' is href, and 'htmlSnippet' is body
            # Based on the output from ali_search_test.py
            title = result.get("hostname","") + ": "+result.get("title", "No Title")
            href = result.get("link", "#")
            body = result.get("mainText") if result.get("mainText") else result.get("snippet","No text")
            
            search_results.append({
                "title": title,
                "href": href,
                "body": body,
            })

        return search_results[:max_results]


