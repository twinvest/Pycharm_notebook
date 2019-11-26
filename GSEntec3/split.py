#google map geocoding function

def geocoding(location):
    import requests
    import json
    url="https://maps.googleapis.com/maps/api/geocode/json?address=%s&key=AIzaSyApdxo0PerumoXiXKY1dQkjC_lavxSAID0" % (location)
    direction=requests.get(url)
    html=direction.text
    geo=json.loads(html)
    if geo['status']=='OK':
        lat=geo['results'][0]['geometry']['location']['lat']
        lng=geo['results'][0]['geometry']['location']['lng']
        lat=str(lat)
        lng=str(lng)
        location=lng+','+lat
        status=geo['status']
        return location, lat, lng, status
    else:
        lat=''
        lng=''
        location=''
        status=geo['status']
        return location, lat, lng, status


# duration function
def direction(start):
    import requests
    import json
    ClientID = '348np9r8nu'
    ClientSecret= 'I5WvQIUd17MweksYkMzHMjUkI9SiXkKa2Gby2s7g'
    headers = {'X-NCP-APIGW-API-KEY-ID': ClientID, 'X-NCP-APIGW-API-KEY': ClientSecret}
    goal= '129.3694633, 35.495691' #gs엔텍
    params={'start':start, 'goal':goal}
    url='https://naveropenapi.apigw.ntruss.com/map-direction/v1/driving'
    direction=requests.get(url, headers=headers, params=params)
    html=direction.text
    direction_parse=json.loads(html)
    # print('code:%s' % direction_parse['code'])
    if direction_parse['code'] == 0:
        duration=direction_parse['route']['traoptimal'][0]['summary']['duration']
        return duration
    else:
        duration=float('NaN')
        return duration


def injapass(inja):
    raw_data = { 'gender': None , 'honey_YN': None, 'child_count': None, 'club': None, 'courier_count': None,
                    'main_dep': None,
                    'age': None,
                    'certificate': None,
                    'duration': None,
                    'position_number_1.0': [0],
                    'position_number_2.0': [0],
                    'position_number_3.0': [0],
                    'position_number_4.0': [0],
                    'position_number_5.0': [0],
                    'position_number_6.0': [0],
                    'area_Chungbuk': [0],
                    'area_Chungnam': [0],
                    'area_Gangwon': [0],
                    'area_Jeonam': [0],
                    'area_Jeonbuk': [0],
                    'area_Kyongbuk': [0],
                    'area_Kyongnam': [0],
                    'area_Metropolitan': [0],
                    'department_Business_Support': [0],
                    'department_Business_management': [0],
                    'department_Design': [0],
                    'department_Financial_management': [0],
                    'department_Production_Management': [0],
                    'department_Production_support': [0],
                    'department_Purchasing': [0],
                    'department_Quality_Management': [0],
                    'department_Research': [0],
                    'department_Sales': [0],
                    'department_Service': [0],
                    'department_Shipping': [0],
                    'department_Technology': [0],
                    'department_Warranty': [0]
                    }

    raw_data['courier_count']=[inja[1]]
    raw_data['age']=[inja[3]]
    if inja[4]=='Chungbuk':
        raw_data['area_Chungbuk']=[1]
    elif inja[4]=='Chungnam':
        raw_data['area_Chungnam']=[1]
    elif inja[4]=='Gangwon':
        raw_data['area_Gangwon']=[1]
    elif inja[4]=='Jeonam':
        raw_data['area_Jeonam']=[1]
    elif inja[4]=='Jeonbuk':
        raw_data['area_Jeonbuk']=[1]
    elif inja[4]=='Kyongbuk':
        raw_data['area_Kyongbuk']=[1]
    elif inja[4]=='Kyongnam':
        raw_data['area_Kyongnam']=[1]
    elif inja[4]=='Metropolitan':
        raw_data['area_Metropolitan']=[1]

    raw_data['child_count']=[inja[5]]

    if inja[6]=='YES':
        raw_data['club']=[1]
    else:
        raw_data['club'] = [0]

    if inja[7]=='남자':
        raw_data['gender'] = [1]
    else:
        raw_data['gender'] = [0]

    if inja[8]=='YES':
        raw_data['honey_YN'] = [1]
    else:
        raw_data['honey_YN']= [0]

    if inja[12]=='부장,소장=1':
        raw_data['position_number_1.0'] = [1]
    elif inja[12]=='직장,차장=2':
        raw_data['position_number_2.0'] = [1]
    elif inja[12]=='반장,과장=3':
        raw_data['position_number_3.0'] = [1]
    elif inja[12]=='대리,조장=4':
        raw_data['position_number_4.0'] = [1]
    elif inja[12]=='주임=5':
        raw_data['position_number_5.0'] = [1]
    elif inja[12]=='사원=6':
        raw_data['position_number_6.0'] = [1]


    if inja[13]=='Production support':
        raw_data['department_Production_support']=[1]
    elif inja[13]=='Purchasing':
        raw_data['department_Purchasing']=[1]
    elif inja[13]=='Production Management':
        raw_data['department_Production_Management']=[1]
    elif inja[13]=='Business management':
        raw_data['department_Business_Support']=[1]
    elif inja[13]=='Quality Management':
        raw_data['department_Quality_Management']=[1]
    elif inja[13]=='Financial management':
        raw_data['department_Financial_management']=[1]
    elif inja[13]=='Sales':
        raw_data['department_Sales']=[1]
    elif inja[13]=='Research':
        raw_data['department_Research']=[1]
    elif inja[13]=='Warranty':
        raw_data['department_Warranty']=[1]
    elif inja[13]=='Shipping':
        raw_data['department_Shipping']=[1]
    elif inja[13]=='Technology':
        raw_data['department_Technology']=[1]
    elif inja[13]=='Service':
        raw_data['department_Service']=[1]

    if inja[14]=='YES':
        raw_data['main_dep']=[1]
    else:
        raw_data['main_dep']=[0]


    start, lati, long, stat = geocoding(' '.join(inja[16:-1]))
    raw_data['duration']=[direction(start)/3600000]

    if inja[15]=='YES':
        raw_data['certificate']=[1]
    else:
        raw_data['certificate']=[0]

    return raw_data

string = ['/home/rlaxodn031/tw_python/inja.py', '5', '2', '27', 'Metropolitan', '3', 'YES', '남자', 'YES', '2016-11-05', '2018-05-31', '3', '직장,차장=2', 'Executives', 'YES', 'YES', '경기도', '고양시', '덕양구', '동산2로 35', '경기대학교']
raw_data = injapass(string)
print(raw_data)