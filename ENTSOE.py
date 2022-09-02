"""
based on https://github.com/EnergieID/entsoe-py/blob/master/entsoe/parsers.py
"""
import pytz
import requests
from bs4 import BeautifulSoup
from time import sleep
import pandas as pd

__title__ = "entsoe-py"
__version__ = "0.1.10"
__author__ = "EnergieID.be"
__license__ = "MIT"

URL = 'https://transparency.entsoe.eu/api'

DOMAIN_MAPPINGS = {
    'AL': '10YAL-KESH-----5',
    'AT': '10YAT-APG------L',
    'BA': '10YBA-JPCC-----D',
    'BE': '10YBE----------2',
    'BG': '10YCA-BULGARIA-R',
    'BY': '10Y1001A1001A51S',
    'CH': '10YCH-SWISSGRIDZ',
    'CZ': '10YCZ-CEPS-----N',
    'DE': '10Y1001A1001A83F',
    'DEp': '10Y1001A1001A63L',
    'DK': '10Y1001A1001A65H',
    'EE': '10Y1001A1001A39I',
    'ES': '10YES-REE------0',
    'FI': '10YFI-1--------U',
    'FR': '10YFR-RTE------C',
    'GB': '10YGB----------A',
    'GB-NIR': '10Y1001A1001A016',
    'GR': '10YGR-HTSO-----Y',
    'HR': '10YHR-HEP------M',
    'HU': '10YHU-MAVIR----U',
    'IE': '10YIE-1001A00010',
    'IT': '10YIT-GRTN-----B',
    'LT': '10YLT-1001A0008Q',
    'LU': '10YLU-CEGEDEL-NQ',
    'LV': '10YLV-1001A00074',
    # 'MD': 'MD',
    'ME': '10YCS-CG-TSO---S',
    'MK': '10YMK-MEPSO----8',
    'MT': '10Y1001A1001A93C',
    'NL': '10YNL----------L',
    'NO': '10YNO-0--------C',
    'PL': '10YPL-AREA-----S',
    'PT': '10YPT-REN------W',
    'RO': '10YRO-TEL------P',
    'RS': '10YCS-SERBIATSOV',
    'RU': '10Y1001A1001A49F',
    'RU-KGD': '10Y1001A1001A50U',
    'SE': '10YSE-1--------K',
    'SI': '10YSI-ELES-----O',
    'SK': '10YSK-SEPS-----K',
    'TR': '10YTR-TEIAS----W',
    'UA': '10YUA-WEPS-----0'
}

TIMEZONE_MAPPINGS = {
    'AL': 'Europe/Tirane',
    'AT': 'Europe/Vienna',
    'BA': 'Europe/Sarajevo',
    'BE': 'Europe/Brussels',
    'BG': 'Europe/Sofia',
    'BY': 'Europe/Minsk',
    'CH': 'Europe/Zurich',
    'CZ': 'Europe/Prague',
    'DE': 'Europe/Berlin',
    'DEp': 'Europe/Berlin',
    'DK': 'Europe/Copenhagen',
    'EE': 'Europe/Talinn',
    'ES': 'Europe/Madrid',
    'FI': 'Europe/Helsinki',
    'FR': 'Europe/Paris',
    'GB': 'Europe/London',
    'GB-NIR': 'Europe/Belfast',
    'GR': 'Europe/Athens',
    'HR': 'Europe/Zagreb',
    'HU': 'Europe/Budapest',
    'IE': 'Europe/Dublin',
    'IT': 'Europe/Rome',
    'LT': 'Europe/Vilnius',
    'LU': 'Europe/Luxembourg',
    'LV': 'Europe/Riga',
    # 'MD': 'MD',
    'ME': 'Europe/Podgorica',
    'MK': 'Europe/Skopje',
    'MT': 'Europe/Malta',
    'NL': 'Europe/Amsterdam',
    'NO': 'Europe/Oslo',
    'PL': 'Europe/Warsaw',
    'PT': 'Europe/Lisbon',
    'RO': 'Europe/Bucharest',
    'RS': 'Europe/Belgrade',
    'RU': 'Europe/Moscow',
    'RU-KGD': 'Europe/Kaliningrad',
    'SE': 'Europe/Stockholm',
    'SI': 'Europe/Ljubljana',
    'SK': 'Europe/Bratislava',
    'TR': 'Europe/Istanbul',
    'UA': 'Europe/Kiev'
}

PSRTYPE_MAPPINGS = {
    'A03': 'Mixed',
    'A04': 'Generation',
    'A05': 'Load',
    'B01': 'Biomass',
    'B02': 'Fossil Brown coal/Lignite',
    'B03': 'Fossil Coal-derived gas',
    'B04': 'Fossil Gas',
    'B05': 'Fossil Hard coal',
    'B06': 'Fossil Oil',
    'B07': 'Fossil Oil shale',
    'B08': 'Fossil Peat',
    'B09': 'Geothermal',
    'B10': 'Hydro Pumped Storage',
    'B11': 'Hydro Run-of-river and poundage',
    'B12': 'Hydro Water Reservoir',
    'B13': 'Marine',
    'B14': 'Nuclear',
    'B15': 'Other renewable',
    'B16': 'Solar',
    'B17': 'Waste',
    'B18': 'Wind Offshore',
    'B19': 'Wind Onshore',
    'B20': 'Other',
    'B21': 'AC Link',
    'B22': 'DC Link',
    'B23': 'Substation',
    'B24': 'Transformer'}


class Entsoe:
    """
    Attributions: Parts of the code for parsing Entsoe responses were copied
    from https://github.com/tmrowco/electricitymap
    """
    def __init__(self, api_key, session=None, retry_count=1, retry_delay=0):
        """
        Parameters
        ----------
        api_key : str
        session : requests.Session
        """
        if api_key is None:
            raise TypeError("API key cannot be None")
        self.api_key = api_key
        if session is None:
            session = requests.Session()
        self.session = session
        self.retry_count = retry_count
        self.retry_delay = retry_delay
        self.gen = 3
        self.forecast_gen = 0
        self.forecast_ws = 0
        self.EF = 0
        self.price = 0
        self.price_q = 0
        self.load = 0


    def base_request(self, params, start, end):
        """
        Parameters
        ----------
        params : dict
        start : pd.Timestamp
        end : pd.Timestamp
        Returns
        -------
        requests.Response
        """
        start_str = self._datetime_to_str(start)
        end_str = self._datetime_to_str(end)

        base_params = {
            'securityToken': self.api_key,
            'periodStart': start_str,
            'periodEnd': end_str
        }
        params.update(base_params)

        error = None
        for _ in range(self.retry_count):
            response = self.session.get(url=URL, params=params)
            try:
                response.raise_for_status()
            except requests.HTTPError as e:
                error = e
                soup = BeautifulSoup(response.text, 'html.parser')
                text = soup.find_all('text')
                if len(text):
                    error_text = soup.find('text').text
                    if 'No matching data found' in error_text:
                        return None
                print("HTTP Error, retrying in {} seconds".format(self.retry_delay))
                sleep(self.retry_delay)
            else:
                return response
        else:
            raise error

    @staticmethod
    def _datetime_to_str(dtm):
        """
        Convert a datetime object to a string in UTC
        of the form YYYYMMDDhh00
        Parameters
        ----------
        dtm : pd.Timestamp
            Recommended to use a timezone-aware object!
            If timezone-naive, UTC is assumed
        Returns
        -------
        str
        """
        if dtm.tzinfo is not None and dtm.tzinfo != pytz.UTC:
            dtm = dtm.tz_convert("UTC")
        fmt = '%Y%m%d%H00'
        ret_str = dtm.strftime(fmt)
        return ret_str

    def query_price(self, country_code, start, end, as_series=False):
        """
        Parameters
        ----------
        country_code : str
        start : pd.Timestamp
        end : pd.Timestamp
        as_series : bool
            Default False
            If True: Return the response as a Pandas Series
            If False: Return the response as raw XML
        Returns
        -------
        str | pd.Series
        """
        domain = DOMAIN_MAPPINGS[country_code]
        params = {
            'documentType': 'A44',
            'in_Domain': domain,
            'out_Domain': domain
        }
        response = self.base_request(params=params, start=start, end=end)
        if response is None:
            print("None")
            return None
        if not as_series:
            print(response.text)
            return response.text
        else:
            from parsers import parse_prices
            series, series_q = parse_prices(response.text)
            series = series.tz_convert(TIMEZONE_MAPPINGS[country_code])
            series_q = series_q.tz_convert(TIMEZONE_MAPPINGS[country_code])

            self.price = series
            self.price_q = series_q

            return series, series_q

    def query_generation_forecast_WS(self, country_code, start, end, as_dataframe=False):
        """
        Parameters
        ----------
        country_code : str
        start : pd.Timestamp
        end : pd.Timestamp
        as_dataframe : bool
            Default False
            If True: Return the response as a Pandas DataFrame
            If False: Return the response as raw XML
        Returns
        -------
        str | pd.DataFrame
        """
        """
        Only Solar, Wind Offshore and Wind Onshore are given
        """
        domain = DOMAIN_MAPPINGS[country_code]
        params = {
            'documentType': 'A69',
            'processType': 'A01',
            'in_Domain': domain,
        }
        response = self.base_request(params=params, start=start, end=end)
        if response is None:
            print("None")
            return None
        if not as_dataframe:
            #print response.text
            return response.text
        else:
            from parsers import parse_generation
            df = parse_generation(response.text)
            df = df.tz_convert(TIMEZONE_MAPPINGS[country_code])
            self.forecast_ws = df
            #print df
            return df


    def query_generation_forecast_all(self, country_code, start, end, as_dataframe=False):
        """
        Parameters
        ----------
        country_code : str
        start : pd.Timestamp
        end : pd.Timestamp
        as_dataframe : bool
            Default False
            If True: Return the response as a Pandas DataFrame
            If False: Return the response as raw XML
        Returns
        -------
        str | pd.DataFrame
        """
        """
           Aggregated day-ahead generation
        """
        domain = DOMAIN_MAPPINGS[country_code]
        params = {
            'documentType': 'A71',
            'processType': 'A01',
            'in_Domain': domain,
        }
        response = self.base_request(params=params, start=start, end=end)
        if response is None:
            return None
        if not as_dataframe:
            print(response.text)
            return response.text
        else:
            from parsers import parse_generation
            df = parse_generation(response.text)
            df = df.tz_convert(TIMEZONE_MAPPINGS[country_code])
            self.forecast_gen = df
            return df


    def query_generation(self, country_code, start, end, as_dataframe=False):
        """
        Parameters
        ----------
        country_code : str
        start : pd.Timestamp
        end : pd.Timestamp
        as_dataframe : bool
            Default False
            If True: Return the response as a Pandas DataFrame
            If False: Return the response as raw XML
        Returns
        -------
        str | pd.DataFrame
        """

        domain = DOMAIN_MAPPINGS[country_code]
        params = {
            'documentType': 'A75',
            'processType': 'A16',
            'in_Domain': domain,
        }
        response = self.base_request(params=params, start=start, end=end)
        if response is None:
            return None
        if not as_dataframe:
            f =  open("myxmlfile.txt", "wb")
            f.write(response.text)
            f.close()
            print(response.text)
            return response.text
        else:
            from parsers import parse_generation
            df = parse_generation(response.text)
            df = df.tz_convert(TIMEZONE_MAPPINGS[country_code])

            self.gen = df

            production = df

            self.EF =  self.data_arrange(production)

            #return df



    def query_installed_generation_capacity(self, country_code, start, end, as_dataframe=False):
        """
        Parameters
        ----------
        country_code : str
        start : pd.Timestamp
        end : pd.Timestamp
        as_dataframe : bool
            Default False
            If True: Return the response as a Pandas DataFrame
            If False: Return the response as raw XML
        Returns
        -------
        str | pd.DataFrame
        """
        domain = DOMAIN_MAPPINGS[country_code]
        params = {
            'documentType': 'A68',
            'processType': 'A33',
            'in_Domain': domain,
        }
        response = self.base_request(params=params, start=start, end=end)
        if response is None:
            return None
        if not as_dataframe:
            return response.text
        else:
            from parsers import parse_generation
            df = parse_generation(response.text)
            df = df.tz_convert(TIMEZONE_MAPPINGS[country_code])
            #print df
            return df


    def query_consumption(self, country_code, start, end, as_dataframe=False):
        domain = DOMAIN_MAPPINGS[country_code]
        params = {
                'documentType': 'A65',
                'processType': 'A16',
                'outBiddingZone_Domain': domain,
        }
        response = self.base_request(params=params, start=start, end=end)
        if response is None:
            print("No consumption forecast")
            return None
        if not as_dataframe:
            #print(response.text)
            return response.text

        else:
            from parsers import parse_loads
            df = parse_loads(response.text)
            df = df.tz_convert(TIMEZONE_MAPPINGS[country_code])
            #print(df)
            self.load = df
            return df
        #else:
        # Grab the error if possible
        #soup = BeautifulSoup(response.text, 'html.parser')
        #text = soup.find_all('text')
        #if len(text):
         #   error_text = soup.find_all('text')[0].prettify()
          #  if 'No matching data found' in error_text: return
           # raise Exception('Failed to get consumption. Reason: %s' % error_text)


    def query_consumption_forecast(self, country_code, start, end, as_dataframe=False):
        domain = DOMAIN_MAPPINGS[country_code]
        params = {
                'documentType': 'A65',
                'processType': 'A01',
                'outBiddingZone_Domain': domain,
        }
        response = self.base_request(params=params, start=start, end=end)
        if response is None:
            print("No consumption forecast")
            return None
        if not as_dataframe:
            #print(response.text)
            return response.text
        else:
            from parsers import parse_loads
            df = parse_loads(response.text)
            df = df.tz_convert(TIMEZONE_MAPPINGS[country_code])

            self.load = df
            #print(df)
            return df
        #else:
        # Grab the error if possible
        #soup = BeautifulSoup(response.text, 'html.parser')
        #text = soup.find_all('text')
        #if len(text):
         #   error_text = soup.find_all('text')[0].prettify()
          #  if 'No matching data found' in error_text: return
           # raise Exception('Failed to get consumption. Reason: %s' % error_text)

    def query_exchange(self, country_from, country_to, start, end, as_dataframe=False):
       in_domain = DOMAIN_MAPPINGS[country_to]
       out_domain = DOMAIN_MAPPINGS[country_from]
       params = {
               'documentType': 'A09',
               'in_Domain': in_domain,
               'out_Domain': out_domain,
       }
       response = self.base_request(params=params, start=start, end=end)
       if response is None:
            print("No exchange data")
            return None
       if not as_dataframe:
            print(response.text)
            return response.text

    def data_arrange(self, generation):
        data_table = generation

        set_columns = generation.columns
        set_gen = ['Fossil Brown coal/Lignite', 'Fossil Coal-derived gas', 'Fossil Gas', 'Fossil Hard coal', 'Fossil Oil',
                   'Fossil Oil shale', 'Fossil Peat', 'Geothermal', 'Hydro Pumped Storage', 'Hydro Run-of-river and poundage',
                   'Hydro Water Reservoir', 'Marine', 'Nuclear', 'Other renewable', 'Solar', 'Waste', 'Wind Offshore', 'Wind Onshore', 'Other', 'Biomass']

        for kind in set_gen:
            if not kind in set_columns:
                data_table[kind] = [0]*(data_table['Fossil Gas'].size)
        return data_table

        #technologies with higher emissions
        #EFt = 0.25*(70.5*(data_table['Other renewable'] + data_table['Biomass]) + 1100.0*(data_table['Fossil Brown coal/Lignite']) + 1001.0*(data_table['Fossil Hard coal'] + data_table['Fossil Coal-derived gas'])
                    #+ 553.0*data_table['Fossil Gas'] + 788.0*data_table['Fossil Oil'] + 45.0*data_table['Geothermal'] + 34.0*data_table['Hydro Pumped Storage'] +
                    #4.5*data_table['Hydro Run-of-river and poundage'] + 9.0*data_table['Hydro Water Reservoir'] + 11.0*data_table['Nuclear'] +
                    #314.0*data_table['Other'] + 56.8*data_table['Solar'] + 690.0*data_table['Waste'] + 14.0*data_table['Wind Offshore'] + 12.0*data_table['Wind Onshore']) / (0.25*data_table.sum(axis=1))

        #technologies with lower emisisons
        EFt = 0.25*(71.0*(data_table['Other renewable'] + data_table['Biomass']) + 820.0*(data_table['Fossil Brown coal/Lignite']) + 800.0*(data_table['Fossil Hard coal'] + data_table['Fossil Coal-derived gas']) +
                   400.0*data_table['Fossil Gas'] + 520.0*data_table['Fossil Oil'] + 45.0*data_table['Geothermal'] + 34.0*data_table['Hydro Pumped Storage'] +
                   4.0*data_table['Hydro Run-of-river and poundage'] + 9.0*data_table['Hydro Water Reservoir'] + 11.0*data_table['Nuclear'] +
                   247.0*data_table['Other'] + 43.0*data_table['Solar'] + 690.0*data_table['Waste'] + 9.0*data_table['Wind Offshore'] + 8.0*data_table['Wind Onshore']) / (0.25*data_table.sum(axis=1))


        return EFt
